# aten.ceil.float

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.ceil.float(a: float) -> int
```

### 1.2 算子功能

对有限标量向正无穷取整，例如 ceil(-1.25) = -1、ceil(1.25) = 2；整数输入保持原值。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `float` | 输入 Python 标量。 |
| `返回值` | `int` | 向上取整结果；本测试输入得到 Python int。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | - | - | - | - | - | √ | √ | √ | - |
| NPU | - | - | - | - | - | - | - | - | √ | √ | √ | - |

#### 2.2.2 Shape 支持

标量输入不涉及 Tensor shape。

### 2.3 特殊限制说明

dtype 用于构造标量，.item() 后不保留 Tensor dtype。辅助 Tensor 固定为 int64、shape 为 (64,)，仅用于承接标量结果，不属于 ceil 输入。标量计算可能被常量折叠，不能根据辅助 Tensor 生成了 Triton 就认定 ceil 本身生成了 kernel。

### 2.4 使用方法

```python
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('dtype', dtypes)
    @parametrize('value', [-1.25, 0.0, 1.25])
    def test_aten_ceil_float(self, dtype, value):
        def op_calc(first_element, scalar):
            y = torch.ops.aten.ceil.float(scalar)
            return first_element + y

        def op_calc_1(first_element, scalar):
            y = torch.ops.aten.ceil.float(scalar)
            return first_element + y + y

        scalar = torch.tensor(value, dtype=getattr(torch, dtype)).item()
        zero_tensor = torch.zeros(64, dtype=torch.int64, device=torch.device("npu"))
        self.execute_inductor_test(
            (64,), dtype, op_calc, op_calc_1, zero_tensor, scalar
        )
```
