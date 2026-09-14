# -*- coding: utf-8 -*-
"""NPU inductor 模式下算子 dtype 支持性验证脚本

用法: 在 NPU 环境中运行
    python check_npu_inductor_support.py
结果按用户给定顺序完整输出到 result.csv（支持 / 不支持 + 完整报错信息）
"""
import os
import csv
import traceback

import torch
try:
    import torch_npu  # noqa: F401  必须在 torch 之后 import, 注册 NPU 后端
except ImportError:
    raise SystemExit("请先安装并初始化 torch_npu（在 NPU 环境中运行本脚本）")

DEVICE = os.environ.get("DEVICE", "npu:0")
DTYPES = {
    "bool": torch.bool,
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
}


def get_op(path):
    """按 'aten.abs.out' 形式的路径取算子"""
    obj = torch.ops
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


def ones(shape, dtype):
    return torch.ones(shape, dtype=DTYPES[dtype], device=DEVICE)


def empty(shape, dtype):
    return torch.empty(shape, dtype=DTYPES[dtype], device=DEVICE)


def run_in_inductor(fn):
    """用 inductor 后端编译并执行，fullgraph=True 保证确实走 inductor 编译"""
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)
    return compiled()


# ---------------- 用例定义：按用户给定顺序 ----------------
CASES = []


def add(op_name, dtype, call):
    CASES.append((op_name, dtype, call))


B = "bool"

# ---------- ① bool 类型支持 ----------
add("aten.as_strided_copy", B, lambda: get_op("aten.as_strided_copy")(ones((8, 8), B), [8, 8], [8, 1], 0))
add("aten.as_strided_copy.default", B, lambda: get_op("aten.as_strided_copy.default")(ones((8, 8), B), [8, 8], [8, 1], 0))
add("aten.as_strided_copy.out", B, lambda: get_op("aten.as_strided_copy.out")(ones((8, 8), B), [8, 8], [8, 1], 0, out=empty((8, 8), B)))
add("aten.diagonal", B, lambda: get_op("aten.diagonal")(ones((8, 8), B), 0, 0, 1))
add("aten.diagonal.default", B, lambda: get_op("aten.diagonal.default")(ones((8, 8), B), 0, 0, 1))
add("aten.diagonal_copy", B, lambda: get_op("aten.diagonal_copy")(ones((8, 8), B), 0, 0, 1))
add("aten.diagonal_scatter", B, lambda: get_op("aten.diagonal_scatter")(ones((8, 8), B), ones((8,), B), 0, 0, 1))
add("aten.diagonal_scatter.default", B, lambda: get_op("aten.diagonal_scatter.default")(ones((8, 8), B), ones((8,), B), 0, 0, 1))
add("aten.split.default", B, lambda: get_op("aten.split.default")(ones((8, 8), B), 4, 0))
add("aten.split_with_sizes", B, lambda: get_op("aten.split_with_sizes")(ones((8, 8), B), [4, 4], 0))
add("aten.split_with_sizes.default", B, lambda: get_op("aten.split_with_sizes.default")(ones((8, 8), B), [4, 4], 0))
add("aten.unbind", B, lambda: get_op("aten.unbind")(ones((4, 8), B), 0))


def _copy_strided():
    # 兼容新老版本命名: prims.copy_strided / prims.copy_stride
    try:
        op = get_op("prims.copy_strided.default")
    except AttributeError:
        op = get_op("prims.copy_stride.default")
    x = ones((8, 8), B)
    return op(x, list(x.stride()))


add("prims.copy_strided.default", B, _copy_strided)
add("aten.as_strided_scatter", B, lambda: get_op("aten.as_strided_scatter")(ones((8, 8), B), ones((4, 4), B), [4, 4], [8, 1], 0))
add("aten.as_strided_scatter.default", B, lambda: get_op("aten.as_strided_scatter.default")(ones((8, 8), B), ones((4, 4), B), [4, 4], [8, 1], 0))
add("aten.resize", B, lambda: get_op("aten.resize")(ones((4, 4), B), [64]))
add("aten.resize.default", B, lambda: get_op("aten.resize.default")(ones((4, 4), B), [64]))
add("aten.abs", B, lambda: get_op("aten.abs")(ones((4, 8), B)))
add("aten.abs.default", B, lambda: get_op("aten.abs.default")(ones((4, 8), B)))
add("aten.abs.out", B, lambda: get_op("aten.abs.out")(ones((4, 8), B), out=empty((4, 8), B)))

# ---------- ② aten._int_mm 各 dtype 支持 ----------
for d in ["float16", "float32", "bfloat16", "int32", "int64", "bool"]:
    add("aten._int_mm", d, lambda d=d: get_op("aten._int_mm")(ones((16, 16), d), ones((16, 16), d)))
    add("aten._int_mm.default", d, lambda d=d: get_op("aten._int_mm.default")(ones((16, 16), d), ones((16, 16), d)))

# ---------- ③ int8 类型支持 ----------
I8 = "int8"
add("aten.sign", I8, lambda: get_op("aten.sign")(ones((4, 8), I8)))
add("aten.sign.default", I8, lambda: get_op("aten.sign.default")(ones((4, 8), I8)))
add("aten.sign.out", I8, lambda: get_op("aten.sign.out")(ones((4, 8), I8), out=empty((4, 8), I8)))

# ---------- ④ int8/int32/int64/bool 类型支持 ----------
for d in ["int8", "int32", "int64", "bool"]:
    add("aten.avg_pool2d", d, lambda d=d: get_op("aten.avg_pool2d")(ones((1, 3, 8, 8), d), [2, 2], [2, 2]))
    add("aten.avg_pool2d.default", d, lambda d=d: get_op("aten.avg_pool2d.default")(ones((1, 3, 8, 8), d), [2, 2], [2, 2]))
    add("aten.avg_pool2d.out", d, lambda d=d: get_op("aten.avg_pool2d.out")(ones((1, 3, 8, 8), d), [2, 2], [2, 2], out=empty((1, 3, 4, 4), d)))
    add("aten.var.out", d, lambda d=d: get_op("aten.var.out")(ones((4, 8), d), [0], correction=0, keepdim=False, out=empty((8,), d)))
    add("aten.var.correction_out", d, lambda d=d: get_op("aten.var.correction_out")(ones((4, 8), d), [0], correction=0, keepdim=False, out=empty((8,), d)))
    add("aten.baddbmm", d, lambda d=d: get_op("aten.baddbmm")(ones((4, 8, 8), d), ones((4, 8, 16), d), ones((4, 16, 8), d)))
    add("aten.reciprocal.out", d, lambda d=d: get_op("aten.reciprocal.out")(ones((4, 8), d), out=empty((4, 8), d)))
    add("aten.rsqrt.out", d, lambda d=d: get_op("aten.rsqrt.out")(ones((4, 8), d), out=empty((4, 8), d)))


def main():
    results = []
    total = len(CASES)
    print(f"设备: {DEVICE}, 共 {total} 个用例\n")

    for i, (op_name, dtype, call) in enumerate(CASES, 1):
        try:
            run_in_inductor(call)
            status, err = "支持", ""
        except Exception:
            status, err = "不支持", traceback.format_exc()
        results.append([i, op_name, dtype, status, err])
        print(f"[{i}/{total}] {op_name}  dtype={dtype}: {status}")

    out_csv = os.environ.get("RESULT_CSV", "result.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["序号", "算子", "dtype", "结果", "报错信息"])
        w.writerows(results)
    print(f"\n结果已写入 {out_csv}")


if __name__ == "__main__":
    main()
