# aten.ne.complex_float

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.ne.complex_float(complex a, float b) -> bool
```

### 1.2 算子功能
比较复数标量 `a` 与浮点标量 `b` 是否不相等，返回 Python `bool`。浮点标量按 Python 复数与实数比较规则参与比较。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `complex` | 复数标量 |
| `b` | `float` | 实数浮点标量 |
| 返回值 | `bool` | `a != b` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| 实现域 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Host Scalar | - | - | - | - | - | - | - | - | √ | √ | √ | - | √ | √ |

仅涉及 Python `complex` 与 `float` 标量，不涉及 Tensor dtype。

#### 2.2.2 Shape 支持
仅支持零维标量，不涉及 Tensor shape。

### 2.3 特殊限制说明
> 返回 Python `bool`。当 `a` 和 `b` 在编译时是可知的 Python 复数/浮点常量时，该比较具备常量折叠条件，output code 可能不保留 `aten.ne.complex_float`；是否实际折叠以测试生成的代码为准。用例使用零值 Tensor 承接结果。

### 2.4 使用方法
```python
    def test_aten_ne_complex_float(self):
        def op_calc(anchor, first, second):
            return anchor + torch.ops.aten.ne.complex_float(first, second)

        def op_calc_1(anchor, first, second):
            y = torch.ops.aten.ne.complex_float(first, second)
            return anchor + y + y

        anchor = torch.zeros((128,), dtype=torch.float32, device='npu')
        self.execute_inductor_test((128,), 'float32', op_calc, op_calc_1,
                                   anchor, complex(1.0, 2.0), 1.0,
                                   op_name='aten.add')
```