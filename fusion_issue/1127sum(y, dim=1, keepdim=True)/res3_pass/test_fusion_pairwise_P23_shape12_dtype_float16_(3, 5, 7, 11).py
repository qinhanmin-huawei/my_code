# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch_npu._C import _npu_getCurrentRawStreamNoWait as get_raw_stream
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch_npu._C import _npu_getCurrentRawStreamNoWait as get_raw_stream
import torch_npu
torch_npu.npu._initialized = torch_npu.npu.is_initialized()
has_initialized = False
import torch_npu._inductor.runtime.triton_heuristics as triton_heuristics

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
import torch_npu
empty_strided_npu = torch_npu._C._empty_strided_npu


# kernel path: /tmp/torchinductor_root/2x/c2xcuzy6qy4hwofw7itdc3uidqazzjzttum6zdtqia3gdnym4tfb.py
# Topologically Sorted Source Nodes: [s, add_tensor], Original ATen: [aten.sum, aten.add]
# Source node to ATen node mapping:
#   add_tensor => add
#   s => sum_1
# Graph fragment:
#   %arg0_1 : Tensor "f16[3, 5, 7, 11][385, 77, 11, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %sum_1 : Tensor "f16[3, 1, 7, 11][77, 77, 11, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%arg0_1, [1], True), kwargs = {})
#   %add : Tensor "f16[3, 5, 7, 11][385, 77, 11, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %sum_1), kwargs = {})
#   return %add
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_poi_fused_add_sum_0 = async_compile.triton('triton_poi_fused_add_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
if not torch_npu.npu.is_initialized() and torch_npu.npu._is_in_bad_fork():
    torch_npu.npu._initialized = True
from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.pointwise(
    size_hints={'z0': 3, 'y1': 5, 'x2': 77}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32', 'X2BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_sum_0', 'mutated_arg_names': [], 'backend_hash': 'B692356DE12406481BF983C430C437EC40749B6CD9E9DBB288C9B81BB5B66BB8', 'split_axis': [0, 1, 2], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 3), ('y1', 5), ('x2', 77)), 'low_dims': {2}, 'num_load': 6, 'num_store': 0, 'num_reduction': 0, 'atomic_add_found': False, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float16, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK', 'X2BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_sum_0(in_ptr0, out_ptr0, z0_numel, y1_numel, x2_numel, Z0BLOCK, Y1BLOCK, X2BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    x2_offset = tl.program_id(2) * X2BLOCK
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (X2BLOCK + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            for loop_x2 in range(loops_x2):
                x2 = x2_offset + (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,:]
                x2_mask = x2 < min(X2BLOCK+x2_offset, x2_numel)
                tmp0 = tl.load(in_ptr0 + (x2 + 77*y1 + 385*z0), x2_mask & y1_mask & z0_mask).to(tl.float32)
                tmp3 = tl.load(in_ptr0 + (x2 + 385*z0), x2_mask & z0_mask).to(tl.float32)
                tmp4 = tl.load(in_ptr0 + (77 + x2 + 385*z0), x2_mask & z0_mask).to(tl.float32)
                tmp6 = tl.load(in_ptr0 + (154 + x2 + 385*z0), x2_mask & z0_mask).to(tl.float32)
                tmp8 = tl.load(in_ptr0 + (231 + x2 + 385*z0), x2_mask & z0_mask).to(tl.float32)
                tmp10 = tl.load(in_ptr0 + (308 + x2 + 385*z0), x2_mask & z0_mask).to(tl.float32)
                tmp1 = tmp0.to(tl.float16)
                tmp2 = tmp1.to(tl.float32)
                tmp5 = tmp3 + tmp4
                tmp7 = tmp5 + tmp6
                tmp9 = tmp7 + tmp8
                tmp11 = tmp9 + tmp10
                tmp12 = tmp11.to(tl.float16)
                tmp13 = tmp12.to(tl.float32)
                tmp14 = tmp2 + tmp13
                tmp15 = tmp14.to(tl.float16)
                tmp16 = tmp15.to(tl.float32)
                tl.store(out_ptr0 + (x2 + 77*y1 + 385*z0), tmp16, x2_mask & y1_mask & z0_mask)
''', device_str='npu')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_npu((3, 5, 7, 11), (385, 77, 11, 1), torch.float16)
            # Topologically Sorted Source Nodes: [s, add_tensor], Original ATen: [aten.sum, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_sum_0:1
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_sum_0.run(arg0_1, buf0, 3, 5, 77, stream=raw_stream0)
            del arg0_1
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((3, 5, 7, 11), (385, 77, 11, 1), device='npu:0', dtype=torch.float16)
    return [arg0_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
