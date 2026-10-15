# aten.eq.float_complex

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.eq.float_complex(float a, complex b) -> bool
```

### 1.2 算子功能

比较浮点标量 `a` 与复数标量 `b` 是否相等（`a == b`），返回 Python `bool`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `float` | 第一个浮点标量。 |
| `b` | `complex` | 第二个复数标量。 |
| `返回值` | `bool` | 相等比较结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 参数 | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `a` | - | - | - | √ | √ | √ | - | - | - |
| `b` | - | - | - | - | - | - | - | √ | √ |

#### 2.2.2 Shape 支持

该算子为纯标量输入，不涉及shape。

### 2.3 特殊限制说明

该接口为纯标量运算；`a` 为 `float` 类型标量，`b` 为 `complex` 类型标量，返回值为 `bool` 类型标量。长度为 128 的全零 Tensor 仅用于承接比较结果并触发 Inductor 融合，不属于该算子的输入。

### 2.4 使用方法

示例中的两个比较参数为 Python 标量字面量，直接作为算子入参传入，取相等值（虚部为 0）使比较结果为 `True`。返回的 Python `bool` 通过与全零 Tensor 相加承接结果并触发 Inductor 融合；全零 Tensor 使用固定 dtype（`float32`），不属于该算子的输入。

```python
    def test_aten_eq_float_complex(self):
        def op_calc(anchor, first, second):
            return anchor + torch.ops.aten.eq.float_complex(first, second)

        def op_calc_1(anchor, first, second):
            y = torch.ops.aten.eq.float_complex(first, second)
            return anchor + y + y

        anchor = torch.zeros((128,), dtype=torch.float32, device='npu')
        self.execute_inductor_test((128,), 'float32', op_calc, op_calc_1,
                                   anchor, 1.0, complex(1.0, 0.0),
                                   op_name='aten.add')
```
