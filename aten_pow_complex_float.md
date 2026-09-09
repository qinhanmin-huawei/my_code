# aten.pow.complex_float

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.pow.complex_float(complex self, float exponent) -> complex
```

### 1.2 算子功能
对复数标量底数和浮点标量指数执行幂运算。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `complex` | 复数底数标量 |
| `exponent` | `float` | 浮点指数标量 |
| 返回值 | `complex` | 复数结果 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | √ | √ |
| NPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | √ | - |

#### 2.2.2 Shape 支持
仅处理零维复数和浮点标量，不涉及 shape 泛化。

### 2.3 特殊限制说明
> 标量算子可能发生常量折叠；用例用零值 Tensor 承接结果。复数底数使用 `complex64`，`complex128` 不纳入考察。

### 2.4 使用方法
```python
    def test_aten_pow_complex_float(self):
        def op_calc(anchor, base, exponent):
            return anchor + torch.ops.aten.pow.complex_float(base, exponent)

        def op_calc_1(anchor, base, exponent):
            y = torch.ops.aten.pow.complex_float(base, exponent)
            return anchor + y + y

        anchor = torch.zeros((128,), dtype=torch.complex64, device='npu')
        self.execute_inductor_test((128,), 'complex64', op_calc, op_calc_1,
                                   anchor, complex(2.0, 1.0), 2.0,
                                   op_name='aten.add')
```