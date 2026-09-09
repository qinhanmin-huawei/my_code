# aten.pow.float

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.pow.float(float self, float exponent) -> float
```

### 1.2 算子功能
对两个浮点标量执行幂运算。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `float` | 浮点底数 |
| `exponent` | `float` | 浮点指数 |
| 返回值 | `float` | 浮点幂运算结果 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | × | - |
| NPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | × | - |

#### 2.2.2 Shape 支持
仅处理零维浮点标量，不涉及 shape 泛化。

### 2.3 特殊限制说明
> 标量结果可能发生常量折叠；用例用零值 Tensor 承接结果。复数类型不适用于本浮点重载。

### 2.4 使用方法
```python
    def test_aten_pow_float(self):
        def op_calc(anchor, base, exponent):
            return anchor + torch.ops.aten.pow.float(base, exponent)

        def op_calc_1(anchor, base, exponent):
            y = torch.ops.aten.pow.float(base, exponent)
            return anchor + y + y

        anchor = torch.zeros((128,), dtype=torch.float32, device='npu')
        self.execute_inductor_test((128,), 'float32', op_calc, op_calc_1,
                                   anchor, 2.0, 3.0,
                                   op_name='aten.add')
```