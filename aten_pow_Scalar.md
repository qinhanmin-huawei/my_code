# aten.pow.Scalar

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.pow.Scalar(Scalar self, Tensor exponent) -> Tensor
```

### 1.2 算子功能
将标量底数与指数张量逐元素进行幂运算，标量底数广播到指数张量。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Scalar` | 标量底数 |
| `exponent` | `Tensor` | 指数张量 |
| 返回值 | `Tensor` | 标量底数与指数张量的幂运算结果 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | √ | √ |
| NPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | - | - |

#### 2.2.2 Shape 支持
`exponent` 支持 1-4 维，输出 shape 与 `exponent` 一致；标量 `self` 不涉及 shape 泛化。

### 2.3 特殊限制说明
> 标量底数按输入张量类型提升；`bool` 不支持，NPU 复数类型不纳入本重载考察。
> 我们的代码和社区的代码都对int类型单独fallback

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_pow_Scalar(self, shape, dtype):
        def op_calc(exponent):
            return torch.ops.aten.pow.Scalar(2, exponent)

        def op_calc_1(exponent):
            y = torch.ops.aten.pow.Scalar(2, exponent)
            return y + y

        exponent = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, exponent)
```

