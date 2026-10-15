#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
import sys
import os
import shutil
import datetime
import unittest
import pandas as pd
from testutils import TestUtils
# 设置当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))
ascend_test_dir = os.path.normpath(os.path.join(current_dir, "../../.."))
sys.path.append(current_dir)
sys.path.append(ascend_test_dir)

from common.json_test_runner import JSONTestRunner
from common.log import getStdoutLogger
LOGGER = getStdoutLogger()


# testfile_keywords = [
#     # "test_aten_as_strided_.py",
#     # "test_aten_as_strided.py",
#     # "test_prims_bitwise_xor.py",
#     # "test_aten_randn.py",
# ]
testfile_keywords = [f for f in os.listdir(current_dir) if f.startswith('test_') and f.endswith('.py')]
def collect_test_cases():
    """收集同目录下所有满足keyword要求的测试用例文件"""
    test_cases = []
    for keyword in testfile_keywords:
        test_cases.extend(find_files_with_keyword(keyword))
    LOGGER.info(f"Test cases loaded num: {len(test_cases)}")
    for test_case in test_cases:
        LOGGER.info(f"Test case load: {test_case}")
    return test_cases


def find_files_with_keyword(keyword):
    """根据关键字查找文件"""
    test_files = []
    with os.scandir(current_dir) as entries:
        files = [entry.name for entry in entries if entry.is_file()]
        for file in files:
            if file == f"{keyword}":
                test_files.append(os.path.join(current_dir, file))
    return test_files


def record_single_op_result(test_report: dict, summary_list: dict, op_record_list: list):
    # 填充2个用例结果表：
    # 1. summary_list 汇总表格：所有op(约100+)的执行结果汇总，包含op总数、pass/fail数等，按用例文件数计（可扩展-体现在邮件）
    # 2. op_record_list 详细表格：所有用例(含泛化，总数1720*N)的执行结果汇总，按泛化shape、dtype进行分类统计
    for test_case in test_report["test_cases"]:
        op_record_list.append({
            "test_id": test_case["test_id"],
            "shape": test_case["param_shape"],
            "dtype": test_case["param_dtype"],
            "status": test_case["status"],
            "error_message": test_case["error_message"],
            "used_func": test_case["used_func"]
        })
    summary_list["total"] += test_report["summary"]["total"]
    summary_list["success"] += test_report["summary"]["success"]
    summary_list["failures"] += test_report["summary"]["failures"]
    summary_list["errors"] += test_report["summary"]["errors"]
    summary_list["skipped"] += test_report["summary"]["skipped"]


def run_tests(test_cases):
    # 创建结果文件
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(current_dir, f"results_{timestamp}")
    # 创建结果文件所在的目录（如果不存在）
    os.makedirs(results_dir, exist_ok=True)
    result_all_table = os.path.join(results_dir, f"atenop_result_table.csv")
    op_record_title = ["test_id", "shape", "dtype", "status", "error_message", "used_func"]
    op_record_list = []
    summary_list = {"total": 0, "success": 0, "failures": 0, "errors": 0, "skipped": 0}
    TestUtils.results_dir = results_dir
    for test_case in test_cases:
        suite = unittest.TestLoader().discover(current_dir, pattern=os.path.basename(test_case))
        json_output_file = os.path.join(results_dir,
                                        test_case.split(os.sep)[-1].replace("test", "result").replace(".py", ".json"))
        runner = JSONTestRunner(verbosity=2, json_output_file=json_output_file)
        # 执行测试用例，收集结果
        test_result, test_report = runner.run(suite)
        record_single_op_result(test_report, summary_list, op_record_list)

    LOGGER.info("*" * 100)  # 分隔符
    LOGGER.info(f"ATEN_OP Test Summary: {summary_list}")
    # 将结果写入 CSV 文件
    df = pd.DataFrame(op_record_list, columns=op_record_title)
    df.to_csv(result_all_table, encoding="utf-8")
    LOGGER.info(f"Test results saved to: {result_all_table}")
    # 移动日志文件
    run_all_file = os.path.join(current_dir, "run_all.log")
    if os.path.exists(run_all_file):
        shutil.copy(run_all_file, results_dir)
        LOGGER.info(f"run_all.log copied to: {results_dir}")
    else:
        LOGGER.warning("run_all.log not found in the current directory.")
    # import pdb;pdb.set_trace()
    ratio_results_file = os.path.join(current_dir, "ratio_results.log")
    if os.path.exists(ratio_results_file):
        shutil.copy(ratio_results_file, results_dir)
        LOGGER.info(f"ratio_results.log copied to: {results_dir}")
    else:
        LOGGER.warning("ratio_results.log not found in the current directory.")

if __name__ == "__main__":
    # 收集所有测试用例
    test_cases = collect_test_cases()
    if not test_cases:
        LOGGER.info("No test cases found! Please check!")
        sys.exit(1)
    # 运行测试
    run_tests(test_cases)
    sys.exit()
