#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
----------------------------------------------------------------------------
Purpose:
Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
----------------------------------------------------------------------------
"""

import os
import sys
import time
import shutil
import re
import ast
import io
import tokenize
import random
import tempfile
import copy
import math
import csv
from torch.testing._internal.common_utils import TestCase
import torch
import torch_npu
import glob
import datetime
# Set by run_accuracy.py before importing test modules.
ACCURACY_ONLY = os.environ.get("ATEN_OP_ACCURACY_ONLY") == "1"
if not ACCURACY_ONLY:
    import pandas as pd

# 设置当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))
ascend_test_dir = os.path.normpath(os.path.join(current_dir, "../../.."))
sys.path.append(ascend_test_dir)
if ACCURACY_ONLY:
    import logging
    LOGGER = logging.getLogger(__name__)
else:
    from common.log import getStdoutLogger
    LOGGER = getStdoutLogger()

if not ACCURACY_ONLY and "TORCH_COMPILE_DEBUG_DIR" in os.environ:
    from torch._inductor import config

    # 关键：让Inductor trace输出跟随Dynamo的debug目录
    config.trace.enabled = True
    # 指向Dynamo生成的 run_* 父目录；inductor会自动追加 /torchinductor 子文件夹
    config.trace.debug_dir = os.environ["TORCH_COMPILE_DEBUG_DIR"]


def profiling_test(fn_triton, args=(), name="97", shape=(), tiling=(), profiler_path=os.path.join(current_dir, 'result_dir')):
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    random_string = ''.join(random.choices("0123456789", k=4))
    profiler_path = os.path.join(profiler_path, f"profiling_{timestamp}_{random_string}")
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
    )
    with torch_npu.profiler.profile(
            activities=[  # torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],
            with_stack=False,
            record_shapes=False,
            profile_memory=False,
            schedule=torch_npu.profiler.schedule(wait=1,
                                                 warmup=1,
                                                 active=30,
                                                 repeat=1,
                                                 skip_first=1),

            experimental_config=experimental_config,

            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiler_path)
    ) as prof:
        # prof.start()
        for i in range(35):
            fn_triton(*args)
            torch.npu.synchronize()
            prof.step()
    # npu_files = glob.glob(os.path.join(profiler_path, '**', 'op_statistic.csv'), recursive=True)
    # latest_npu_file = sorted(npu_files, key=os.path.getmtime)[-1]
    npu_dirs = glob.glob(os.path.join(profiler_path, '**', 'ASCEND_PROFILER_OUTPUT'), recursive=True)
    latest_npu_dir = sorted(npu_dirs, key=os.path.getmtime)[-1]
    latest_npu_file = os.path.join(latest_npu_dir, 'op_statistic.csv')
    if not os.path.exists(latest_npu_file):
        LOGGER.error(f"{latest_npu_file} not found")
        return -1
    sum_avg = 0
    with open(latest_npu_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sum_avg += float(row['Total Time(us)'])
    sum_avg = sum_avg / 30
    print(f"========================profile data is {sum_avg}")
    return sum_avg


class TestUtils(TestCase):
    result_dir = None
    A3_ub_size = 98304 * 2  # 192KB
    A5_ub_size = 126976 * 2  # 248KB

    _test_ints_dtypes = ['int8', 'int32', 'int64']
    _test_ints_bool_dtypes = ['int8', 'int32', 'int64', 'bool']
    _test_float_dtypes = ['float16', 'bfloat16', 'float32']
    _test_complex_dtypes = ['complex64', 'complex128']
    # _test_dtypes = ['float32']
    _test_dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']
    _test_dtypes_without_bool = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    #_pointwise_demo_shapes =[(32,),(32,32),(32,32,32),(4,8,16,32)]
    _pointwise_demo_shapes = [(4,),(4,8),(4,8,16),(4,8,16,32)]
    _pointwise_demo_shapes_2 = [(4,8),(4,8,16),(4,8,16,32)]
    _pointwise_demo_shapes_mm = [(128,128)]


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_shape = None
        self.param_dtype = None
        self.error_message = None
        self.used_func = None
        self.status = 'failure'
        self.ratio_result = 'UNKNOW'
        self.optype = '?'
        self.ratio = None
        self.compare_target = 'eager'
        if ACCURACY_ONLY:
            return
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(current_dir, f"results_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir
        self.result_all_table = os.path.join(results_dir, f"atenop_result_table.csv")
        self.op_record_title = ["test_id", "shape", "dtype", "status", "error_message", "used_func","ratio_result","ratio"]
        self._compile_attempts = []
        self._current_compile_debug_dir = None
        self._selected_output_codes = []
        df = pd.DataFrame(columns=self.op_record_title)
        df.to_csv(self.result_all_table, encoding="utf-8", index = False)

    @staticmethod
    def _safe_filename(value):
        return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_") or "unknown"

    def _compile_debug_root(self):
        configured_root = os.environ.get("TORCH_COMPILE_DEBUG_DIR")
        if configured_root:
            return os.path.abspath(os.path.expanduser(configured_root))
        return os.path.join(current_dir, "torch_compile_debug")

    def _prepare_compile_debug_dir(self, attempt_name):
        debug_root = self._compile_debug_root()
        os.makedirs(debug_root, exist_ok=True)
        prefix = "{}_{}_".format(
            self._safe_filename(self.id()),
            self._safe_filename(attempt_name),
        )
        debug_dir = tempfile.mkdtemp(prefix=prefix, dir=debug_root)

        from torch._inductor import config
        config.trace.enabled = True
        config.trace.debug_dir = debug_dir

        self._current_compile_debug_dir = debug_dir
        self._selected_output_codes = []
        self._compile_attempts.append({
            "name": attempt_name,
            "debug_dir": debug_dir,
            "shape": self.param_shape,
            "dtype": self.param_dtype,
        })
        LOGGER.info(f"Inductor output for this compile is isolated in: {debug_dir}")
        return debug_dir

    @staticmethod
    def _output_codes_in(debug_dir):
        if not debug_dir or not os.path.isdir(debug_dir):
            return []

        output_codes = []
        for root, _, files in os.walk(debug_dir):
            if "output_code.py" in files:
                output_codes.append(os.path.join(root, "output_code.py"))
        return sorted(output_codes)

    def _cleanup_compile_debug_dirs(self):
        for attempt in self._compile_attempts:
            debug_dir = attempt["debug_dir"]
            if os.path.isdir(debug_dir):
                shutil.rmtree(debug_dir)
                LOGGER.info(f"Removed compile-local debug directory: {debug_dir}")

    def find_file(self, directory, filename):
        for root, dirs, files in os.walk(directory):
            if filename in files:
                return os.path.join(root, filename)
        return None

    @staticmethod
    def clear_env(mode='all'):
        def del_file(path):
            if not os.path.exists(path):
                LOGGER.warning(f"{path} 路径不存在")
            else:
                shutil.rmtree(path)
                LOGGER.info(f"clear env done: {path}")
                time.sleep(0.5)
        try:
            if mode == 'triton_cache' or mode == 'all':
                for file_path in ["/tmp/torchinductor_root", "~/.triton/dump", "~/.triton/cache"]:
                    del_file(file_path)
            if mode == 'torch_compile_debug' or mode == 'all':
                file_path = "torch_compile_debug"  # """先转存output_code，后删除"""
                del_file(file_path)
        except Exception as e:
            LOGGER.error(f"Failed to clear env: {file_path} {e}")

    @staticmethod
    def _generate_tensor(shape, dtype, floatPOSIFLAG=0, min=0, max=0):
        if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
            if min != 0 or max != 0:
                tmp = torch.empty(size=shape, dtype=eval('torch.' + dtype), device=torch.device("npu"))
                torch.nn.init.uniform_(tmp, a=min, b=max)  # 直接指定范围
                return tmp
            if floatPOSIFLAG:
                return 1000 * torch.rand(size=shape, dtype=eval('torch.' + dtype), device=torch.device("npu"))
            else:
                return torch.randn(size=shape, dtype=eval('torch.' + dtype), device=torch.device("npu")) * 2000
        elif dtype == 'int32' or dtype == 'int64':
            high_val = 2000
            low_val = 0
            if max != 0:
                high_val = math.floor(max)
            if min != 0:
                low_val = math.ceil(min)
            return torch.randint(low=low_val, high=high_val, size=shape, dtype=eval('torch.' + dtype),
                                 device=torch.device("npu"))
        elif dtype == 'int8':
            high_val = 10
            low_val = 0
            if max != 0:
                high_val = math.floor(max)
            if min != 0:
                low_val = math.ceil(min)
            return torch.randint(low=low_val, high=high_val, size=shape, dtype=eval('torch.' + dtype),
                                 device=torch.device("npu"))
        elif dtype == 'bool':
            return torch.randint(low=0, high=2, size=shape, device=torch.device("npu")).bool()
        else:
            raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

    @staticmethod
    def _generate_random_scalar(dtype, floatPOSIFLAG=0):
        if dtype in ['float32', 'float16', 'bfloat16']:
            if floatPOSIFLAG:
                return 1000 * torch.rand(size=(), dtype=eval('torch.' + dtype), device=torch.device("npu")).item()
            else:
                return (torch.randn(size=(), dtype=eval('torch.' + dtype), device=torch.device("npu")) * 2000).item()
        elif dtype in ['int32', 'int64']:
            return torch.randint(low=0, high=2000, size=(), dtype=eval('torch.' + dtype),
                                 device=torch.device("npu")).item()
        elif dtype == 'bool':
            return torch.randint(low=0, high=2, size=(), device=torch.device("npu")).bool().item()
        elif dtype in ['int8', 'int16']:
            return torch.randint(low=-128, high=127, size=(), dtype=eval('torch.' + dtype),
                                 device=torch.device("npu")).item()
        else:
            raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

    @classmethod
    def setUpClass(cls):
        if ACCURACY_ONLY:
            super().setUpClass()
            return
        LOGGER.info(
            f"ratio_result=================================== CASE {cls.__name__} START ===================================")
        cls.clear_env()
        cls.success_count = 0  # 统计精度成功用例数
        cls.profiling_success_count = 0  # 统计性能成功用例数
        cls.total_count = 0     # 统计总用例数

    @classmethod
    def tearDownClass(cls):
        if ACCURACY_ONLY:
            super().tearDownClass()
            return
        LOGGER.info(f"=================================== CASE {cls.__name__} END ===================================")
        LOGGER.info(f"accuracy_result:{cls.success_count}/{cls.total_count},ratio_result:{cls.profiling_success_count}/{cls.total_count}")

    def setUp(self):
        LOGGER.info(f"=-=-=-=-=-=-=-=-=-=-=-=- TestCase_ID: {self.id()} -=-=-=-=-=-=-=-=-=-=-=-=")
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        if ACCURACY_ONLY:
            super().tearDown()
            return
        try:
            super().tearDown()
            LOGGER.info(
                f"param_shape: {self.param_shape}, param_dtype: {self.param_dtype}, error_message: {self.error_message}, used_func: {self.used_func}")
            op_record_list=[{
                "test_id": self.id(),
                "shape": self.param_shape,
                "dtype": self.param_dtype,
                "status": self.status,
                "error_message": self.error_message,
                "used_func": self.used_func,
                "ratio_result": self.ratio_result,
                "ratio": self.ratio}]
            df = pd.DataFrame(op_record_list, columns=self.op_record_title)
            df.to_csv(self.result_all_table, mode='a', header=False, encoding="utf-8", index=False)

            target_dir = os.path.join(self.results_dir, "output_code")
            os.makedirs(target_dir, exist_ok=True)
            archived_count = 0
            def_name = self._safe_filename(self.id().split(".")[-1])
            for attempt_index, attempt in enumerate(self._compile_attempts):
                output_codes = self._output_codes_in(attempt["debug_dir"])
                for output_index, source_file_path in enumerate(output_codes):
                    parts = [
                        def_name,
                        self._safe_filename(attempt["name"]),
                        self._safe_filename(attempt["dtype"]),
                        self._safe_filename(attempt["shape"]),
                        str(attempt_index),
                        str(output_index),
                    ]
                    target_file_path = os.path.join(target_dir, "_".join(parts) + ".py")
                    shutil.copy2(source_file_path, target_file_path)
                    archived_count += 1
                    LOGGER.info(f"Archived own output_code: {source_file_path} -> {target_file_path}")

            if archived_count == 0:
                LOGGER.error("No output_code.py was generated by this test case.")
        finally:
            self._cleanup_compile_debug_dirs()
            # Keep the original cross-test cache isolation without deleting
            # another test's debug output root.
            self.clear_env("triton_cache")
            LOGGER.info(f"Test results saved to: {self.result_all_table}")

    def _execute_accuracy_test(self, shape, dtype, func, args, op_name):
        """Run the original operator once per mode, with independent inputs."""
        from torch.utils._pytree import tree_leaves, tree_map

        self.param_shape = str(shape)
        self.param_dtype = dtype
        self.op_name = op_name
        self.used_func = "func"
        self.compare_target = 'eager'
        eager_args = copy.deepcopy(args)
        inductor_args = copy.deepcopy(args)
        try:
            eager_result = func(*eager_args)
        except Exception as error:
            self.error_message = "eager failure"
            print(f"eager failure: {error}")
            try:
                cpu_args = self.move_tensors_to_cpu(copy.deepcopy(args))
                self.compare_target = 'cpu'
                eager_result = func(*cpu_args)
            except Exception as cpu_error:
                raise RuntimeError(
                    f"eager execution and cpu execution failed: {cpu_error}"
                ) from cpu_error
        try:
            # Do not silently turn compiler errors into an eager-only pass.
            with torch._dynamo.config.patch(suppress_errors=False):
                compiled_func = torch.compile(func, backend="inductor")
                inductor_result = compiled_func(*inductor_args)
        except Exception as error:
            raise RuntimeError(f"inductor execution failed: {error}") from error

        result_dtype = next(
            (str(value.dtype).removeprefix("torch.")
             for value in tree_leaves(eager_result) if isinstance(value, torch.Tensor)),
            dtype,
        )
        if self.compare_target == 'cpu':
            inductor_result = tree_map(
                lambda value: value.cpu() if isinstance(value, torch.Tensor) else value,
                inductor_result,
            )
        self.accuracy_validate(eager_result, inductor_result, result_dtype)
        self.status = "success"

    def execute_inductor_test(self, shape, dtype, func, func_plus, *args, op_name=None):
        if ACCURACY_ONLY:
            return self._execute_accuracy_test(shape, dtype, func, args, op_name)
        def get_res_dtype(std_result):
            # 精度校验
            try:
                res_dtype = str(std_result.dtype).replace('torch.', '')
            except AttributeError:
                try:
                    res_dtype = str(std_result[0].dtype).replace('torch.', '')
                except (AttributeError, IndexError, TypeError):
                    # 如果还报错，中断执行
                    raise RuntimeError("无法获取数据类型")
            return res_dtype
        
        self.__class__.total_count += 1
        self.op_name = op_name
        # 在eager、inductor模式下执行op方法，并进行相关校验
        if not self.param_dtype:
            self.param_dtype = dtype
        self.param_shape = str(shape)
        args1 = copy.deepcopy(args)
        args2 = copy.deepcopy(args)
        # ========================================用func测试 ====================================================

        self.compare_target = 'eager'
        self.used_func = "func"
        try:
            std_result = func(*args1)
        except Exception as e:
            self.error_message = "eager failure"
            print(f"eager failure: {e}")
            try:
                args_cpu = self.move_tensors_to_cpu(copy.deepcopy(args))
                self.compare_target = 'cpu'
                std_result = func(*args_cpu)
            except Exception as cpu_error:
                raise AssertionError(
                    f"eager failure and cpu failure: {cpu_error}"
                ) from cpu_error
        self._prepare_compile_debug_dir("func")
        compiled_op_calc = torch.compile(func, backend="inductor")
        try:
            inductor_result = compiled_op_calc(*args2)
        except Exception as e:
            self.error_message = "inductor failure"
            raise AssertionError(f"inductor failure: {e}") from e
        # inductor 功能校验

        self.check_compile_kernel()
        # ========================================用func_plus测试 =================================================
        if self.error_message == "FALLBACK" or self.error_message == "[failure]no output_code generated":
            self.compare_target = 'eager'
            args1 = copy.deepcopy(args)
            args2 = copy.deepcopy(args)
            # 在eager、inductor模式下执行op方法，并进行相关校验
            self.used_func = "func_plus"
            try:
                std_result = func_plus(*args1)
                # pass
            except Exception as e:
                self.error_message = "eager failure"
                print(f"eager failure: {e}")
                try:
                    args_cpu = self.move_tensors_to_cpu(copy.deepcopy(args))
                    self.compare_target = 'cpu'
                    std_result = func_plus(*args_cpu)
                except Exception as cpu_error:
                    raise AssertionError(
                        f"eager failure and cpu failure: {cpu_error}"
                    ) from cpu_error
            self._prepare_compile_debug_dir("func_plus")
            compiled_op_calc = torch.compile(func_plus, backend="inductor")
            try:
                inductor_result = compiled_op_calc(*args2)
            except Exception as e:
                self.error_message = "inductor failure"
                raise AssertionError(f"inductor failure: {e}") from e
            # inductor 功能校验
            self.check_compile_kernel()
            if self.error_message == "FALLBACK" or self.error_message == "[failure]no output_code generated":
                raise AssertionError(f"CAN NOT lowering!!!!!!!!!!!")
            # 精度校验
        print(f"std_result:\n{std_result}")
        print(f"inductor_result:\n{inductor_result}")
        res_dtype = get_res_dtype(std_result)
        # self.accuracy_validate(std_result, inductor_result, res_dtype)
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        if self.used_func == "func":
            last_func = func
        elif self.used_func == "func_plus":
            last_func = func_plus
        else:
            raise AssertionError(f"used_func : {self.used_func} , is invalid")

        if self.compare_target == 'cpu':
            from torch.utils._pytree import tree_map

            inductor_result_cpu = tree_map(
                lambda value: value.cpu() if isinstance(value, torch.Tensor) else value,
                inductor_result,
            )
            try:
                self.accuracy_validate(std_result, inductor_result_cpu, res_dtype)
            except AssertionError as e:
                self.error_message = "Precision verification failed"
                raise AssertionError(
                    f"Inductor-CPU Precision verification failed: {e}"
                ) from e
            self.status = 'success'
            self.__class__.success_count += 1
            self.ratio_result = 'PASS'
            self.__class__.profiling_success_count += 1
            return

        self.compare_with_cpu(std_result, inductor_result, res_dtype, last_func, *args)
        self.status = 'success'
        self.__class__.success_count += 1

        args1 = copy.deepcopy(args)
        args2 = copy.deepcopy(args)
        eager_perfdata = profiling_test(last_func, args1)
        inductor_perfdata = profiling_test(compiled_op_calc, args2)
        ratio = eager_perfdata / inductor_perfdata
        #Cube/Vector算子性能0.7xAsc、CV算子性能0.5xAsc
        if eager_perfdata == -1 or inductor_perfdata == -1:
            ratio = -1
            self.ratio_result = 'ERROR'
        elif (self.optype == 'CUBE' or self.optype == 'VV' ) and ratio < 0.7:
            self.ratio_result = 'FAIL'
        elif self.optype == 'CV' and ratio < 0.5:
            self.ratio_result = 'FAIL'
        elif self.optype == '?' and ratio < 0.7:
            self.ratio_result = 'FAIL'
        else:
            self.ratio_result = 'PASS'
            self.__class__.profiling_success_count += 1
        LOGGER.info(
            f"case:{self.id}, \tparam_dtype: {self.param_dtype}, \tparam_shape: {self.param_shape}, \teager: {eager_perfdata}, \tinductor:{inductor_perfdata}, \tratio: {ratio}, \tratio_result: {self.ratio_result}, \terror_message: {self.error_message}")
        
        self.ratio = ratio

    def remove_comments_tokenize(self, code):
        result = []

        for line in code.split('\n'):
            quote_char = None
            escape = False

            # 寻找注释位置
            for i, char in enumerate(line):
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif quote_char is None and char in '\'"':
                    quote_char = char
                elif char == quote_char:
                    quote_char = None
                elif quote_char is None and char == '#':
                    line = line[:i].rstrip()
                    break

            # 添加非空行
            if line.strip() or (line and not line.strip()):
                result.append(line)

        return '\n'.join(result)

    def detect_pattern_in_code(self, op_pattern, code):
        """
        使用正则表达式检测代码中是否包含指定模式

        Args:
            op_pattern (str): 正则表达式模式
            code (str): 要检测的代码字符串

        Returns:
            bool: 如果检测到返回True，否则返回False
        """
        try:
            # 编译正则表达式（提高性能）
            pattern = re.compile(op_pattern)
            # 在代码中搜索模式
            match = pattern.search(code)
            # 如果找到匹配则返回True
            return match is not None
        except re.error as e:
            LOGGER.error(f"Regular expression error: {e}")
            return False

    def show_triton_kernel(self, content):
        # 打印部分 kernel 内容，方便定位
        LOGGER.info("Triton kernel showof:")
        LOGGER.info("-" * 100)
        # 分行打印内容，每行10个字符为一组便于阅读
        lines = content.split('\n')
        triton_index = -1
        for i, line in enumerate(lines):
            if '@triton.jit' in line:
                triton_index = i
                break

        end_line = min(triton_index + 15, len(lines))
        for i in range(triton_index, end_line):
            LOGGER.info(f"{lines[i]}")
        LOGGER.info("-" * 100)

    def remove_comments_tokenize(self, code):
        result = []

        for line in code.split('\n'):
            quote_char = None
            escape = False

            # 寻找注释位置
            for i, char in enumerate(line):
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif quote_char is None and char in '\'"':
                    quote_char = char
                elif char == quote_char:
                    quote_char = None
                elif quote_char is None and char == '#':
                    line = line[:i].rstrip()
                    break

            # 添加非空行
            if line.strip() or (line and not line.strip()):
                result.append(line)

        return '\n'.join(result)

    def get_op_pattern(self):
        if self.op_name == None:
            method_name = self.id().split(".")[2]
            func_name = method_name.split("test_")[1].split("_shape")[0]
            op_parts = [p for p in func_name.split('_') if p != ""]
            op_pattern = r'torch\.ops\._{0,}' + re.escape(op_parts[0]) + r'_{0,}' + r'.{0,}' + r'_{0,}' + re.escape(
                op_parts[1])
        else:
            op_name = self.op_name
            op = ".".join(op_name.split('.')[:4])
            op_pattern = re.escape(op)
        return op_pattern

    def detect_pattern_in_code(self, op_pattern, code):
        """
        使用正则表达式检测代码中是否包含指定模式

        Args:
            op_pattern (str): 正则表达式模式
            code (str): 要检测的代码字符串

        Returns:
            bool: 如果检测到返回True，否则返回False
        """
        try:
            # 编译正则表达式（提高性能）
            pattern = re.compile(op_pattern)
            # 在代码中搜索模式
            match = pattern.search(code)
            # 如果找到匹配则返回True
            return match is not None
        except re.error as e:
            LOGGER.error(f"Regular expression error: {e}")
            return False

    @staticmethod
    def _call_function_sources(content):
        try:
            tree = ast.parse(content)
        except SyntaxError as error:
            LOGGER.error(f"Failed to parse output_code.py: {error}")
            return []

        lines = content.splitlines()
        call_sources = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "call":
                end_lineno = getattr(node, "end_lineno", node.lineno)
                call_sources.append("\n".join(lines[node.lineno - 1:end_lineno]))
        return call_sources

    def check_compile_kernel(self):
        # Only inspect files produced under the directory assigned to this compile.
        output_codes = self._output_codes_in(self._current_compile_debug_dir)
        self._selected_output_codes = output_codes
        LOGGER.info(
            f"Current compile output_code files in {self._current_compile_debug_dir}: {output_codes}")
        if not output_codes:
            self.error_message = "[failure]no output_code generated"
            return

        has_triton_jit = False
        has_call_function = False
        fallback_files = []
        op_pattern = self.get_op_pattern()

        for file_path in output_codes:
            with open(file_path, 'r', encoding='utf-8') as output_file:
                content = output_file.read()

            if any(line.strip() == "@triton.jit" for line in content.splitlines()):
                has_triton_jit = True

            call_sources = self._call_function_sources(content)
            has_call_function = has_call_function or bool(call_sources)
            if any(
                    self.detect_pattern_in_code(
                        op_pattern, self.remove_comments_tokenize(call_source))
                    for call_source in call_sources):
                fallback_files.append(file_path)

        if fallback_files or not has_triton_jit or not has_call_function:
            self.error_message = "FALLBACK"
            LOGGER.error(
                f"FALLBACK - {self.used_func}; fallback_files={fallback_files}, "
                f"has_triton_jit={has_triton_jit}, has_call_function={has_call_function}")
        else:
            self.error_message = "kernel generated"

    def accuracy_validate(self, eager_result, inductor_result, dtype, equal_nan=True):
        if not self.param_dtype:
            self.param_dtype = dtype
        if dtype == 'float64':
            self.rtol, self.atol = 1e-8, 1e-10
        elif dtype == 'float32':
            self.rtol, self.atol = 1e-5, 1e-5
        elif dtype == 'float16':
            self.rtol, self.atol = 1e-3, 1e-5
        elif dtype == 'bfloat16':
            self.rtol, self.atol = 1.6e-2, 1e-5
        elif dtype == 'uint8' or dtype == 'int8':
            self.rtol, self.atol = 1e-2, 1e-3
        else:
            self.rtol, self.atol = 0, 0
        try:
            if dtype == 'bool':
                torch.testing.assert_close(~eager_result, ~inductor_result, equal_nan=equal_nan, rtol=self.rtol, atol=self.atol)
            else:
                torch.testing.assert_close(eager_result, inductor_result, equal_nan=equal_nan, rtol=self.rtol, atol=self.atol)
        except Exception as e:
            # self.error_message = "Precision verification failed"
            
            print(f"==============={self.compare_target}和compile精度不通过:{self.id}=====================================")
            print(f"{e}")
            print("===============================================================================================")
            raise AssertionError(f"inductor vs {self.compare_target} Precision verification failed: {e}") from e


    def accuracy_validate_native_dropout(self, eager_result, inductor_result, equal_nan, rtol, atol):
        eager_active_mask = (eager_result != 0)
        inductor_active_mask = (inductor_result != 0)

        combined_mask = eager_active_mask & inductor_active_mask
        if combined_mask.any():
            torch.testing.assert_close(
                eager_result[combined_mask],
                inductor_result[combined_mask],
                equal_nan=equal_nan,
                rtol=rtol, atol=atol
            )

        actual_dropped_eager = (~eager_active_mask).float().mean().item()
        actual_dropped_ind = (~inductor_active_mask).float().mean().item()
        torch.testing.assert_close(actual_dropped_eager, actual_dropped_ind, rtol=rtol, atol=atol)

    def compare_with_cpu(self, eager_result, inductor_result, dtype, func, *args):
        try:
            self.accuracy_validate(eager_result, inductor_result, dtype)
        except AssertionError:
            print("compare with cpu")
            try:
                args_cpu = self.move_tensors_to_cpu(args)
                cpu_result = func(*args_cpu)
                try:
                    print(f"inductor vs cpu")
                    torch.testing.assert_close(cpu_result, inductor_result.cpu(), equal_nan=True, rtol=self.rtol, atol=self.atol)
                except AssertionError as e:
                    print(f"inductor vs cpu fail")
                    print(f"{e}")
                try:
                    print(f"eager vs cpu")
                    torch.testing.assert_close(cpu_result, eager_result.cpu(), equal_nan=True, rtol=self.rtol, atol=self.atol)
                except AssertionError as e:
                    print(f"eager vs cpu fail")
                    print(f"{e}")
                
                eager_cpu_diff = torch.abs(eager_result.cpu() - cpu_result).float().mean().item()
                inductor_cpu_diff  = torch.abs(inductor_result.cpu() - cpu_result).float().mean().item()
                self.assertLessEqual(inductor_cpu_diff , eager_cpu_diff)
            except AssertionError as e:
                self.error_message = "Precision verification failed"
                raise AssertionError(f"GPU-CPU NPU-CPU Precision verification failed: {e}") from e

    def move_tensors_to_cpu(self, args):
        args_cpu = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                args_cpu.append(arg.cpu())
            elif isinstance(arg, list):
                args_cpu.append(self.move_tensors_to_cpu(arg))
            else:
                args_cpu.append(arg)
        return args_cpu
