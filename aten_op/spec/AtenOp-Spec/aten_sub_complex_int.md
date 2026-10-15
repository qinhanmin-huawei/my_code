# aten.sub.complex_int

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sub.complex_int(complex a, int b) -> complex
```

### 1.2 算子功能

计算两个 Python 数值标量之差 `a - b`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `complex` | 被减数。 |
| `b` | `int` | 减数。 |
| `返回值` | `complex` | 标量差值。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | √ | √ | √ | - | - | - | - | √ | √ |
| NPU | √ | √ | √ | - | - | - | - | √ | √ |

#### 2.2.2 Shape 支持

该算子为纯标量输入，不涉及shape。

### 2.3 特殊限制说明

该接口为纯标量运算；`a` 为 `complex` 类型标量，`b` 为 `int` 类型标量，返回值为 `complex` 类型标量。用例按 `complex64` 和 `complex128` 精度构造 `a`；长度为 64 的全零 Tensor 仅用于触发 Inductor 融合，不属于该算子的输入。

### 2.4 使用方法

示例中的 dtype 参数用于构造标量输入，再通过 `.item()` 转为 Python 标量；Python 标量不保留 Tensor dtype。混合类型重载分别遍历两个输入的合法 dtype。辅助融合的全零 Tensor 使用固定 dtype，不参与输入 dtype 的遍历。

```python
    first_dtypes = ['complex64', 'complex128']
    second_dtypes = ['int8', 'int32', 'int64']

    @parametrize('first_dtype', first_dtypes)
    @parametrize('second_dtype', second_dtypes)
    def test_aten_sub_complex_int(self, first_dtype, second_dtype):
        dtype = f"{first_dtype},{second_dtype}"

        def op_calc(first_element, first_scalar, second_scalar):
            y = torch.ops.aten.sub.complex_int(first_scalar, second_scalar)
            result = first_element + y
            return result

        def op_calc_1(first_element, first_scalar, second_scalar):
            y = torch.ops.aten.sub.complex_int(first_scalar, second_scalar)
            result = first_element + y + y
            return result

        zero_tensor = torch.zeros(64, dtype=torch.complex64, device=torch.device("npu"))
        first_scalar = torch.tensor(complex(3.0, 0.5), dtype=getattr(torch, first_dtype)).item()
        second_scalar = torch.tensor(1, dtype=getattr(torch, second_dtype)).item()
        self.execute_inductor_test(
            (64,), dtype, op_calc, op_calc_1, zero_tensor, first_scalar, second_scalar
        )
```
