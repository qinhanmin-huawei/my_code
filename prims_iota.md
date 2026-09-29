# prims.iota

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.prims.iota(
    SymInt length, *, SymInt start, SymInt step,
    ScalarType dtype, Device device, bool requires_grad
) -> Tensor
```

### 1.2 算子功能

生成一维等差整数序列：`y[i] = start + i * step`，其中 `0 <= i < length`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `length` | `SymInt` | 序列长度 |
| `start` | `SymInt` | 起始值 |
| `step` | `SymInt` | 非零步长，可为负数 |
| `dtype` | `ScalarType` | 输出整数 dtype |
| `device` | `Device` | 输出设备 |
| `requires_grad` | `bool` | 当前整数输出要求 False |
| 返回值 | `Tensor` | 一维整数序列 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | dtype（输出） | - | √ | - | - | - | √ | - | √ | × | × | × | × |
| NPU | dtype（输出） | - | √ | - | - | - | √ | - | √ | × | × | × | × |

NPU Eager 路径的 `aclnnArange` 不支持 int8 输出；Inductor 路径支持，结果与 CPU Eager 一致。

#### 2.2.2 Shape 支持

输出形状为 `(length,)`；`length=0` 时返回空张量。

### 2.3 特殊限制说明

> `length >= 0`、`step != 0`，`requires_grad=False`；当前规格要求序列值可由目标整数 dtype 表示。

空序列不要求生成计算 kernel，不作为独立设备 kernel 性能用例。

### 2.4 使用方法

辅助张量仅提供长度、dtype 和 device，其元素值不参与序列生成。框架示例使用非空 int32、int64 序列。

```python
    dtypes = ['int32', 'int64']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    lengths = list(dict.fromkeys(shape[-1] for shape in shapes))
    cases = [dict(length=length, start=0, step=1) for length in lengths]
    if lengths:
        cases.append(dict(length=lengths[0], start=lengths[0] - 1, step=-1))

    @parametrize('case', cases)
    @parametrize('dtype', dtypes)
    def test_prims_iota(self, case, dtype):
        shape = (case['length'],)
        start = case['start']
        step = case['step']

        def op_calc(tensor, start, step):
            y = torch.ops.prims.iota(tensor.numel(), start=start, step=step, dtype=tensor.dtype, device=tensor.device, requires_grad=False)
            return y

        def op_calc_1(tensor, start, step):
            y = torch.ops.prims.iota(tensor.numel(), start=start, step=step, dtype=tensor.dtype, device=tensor.device, requires_grad=False)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, start, step)
```
