# aten.pow.Tensor_Scalar

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
```

### 1.2 算子功能
对输入张量逐元素执行标量指数幂运算。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 幂运算底数张量 |
| `exponent` | `Scalar` | 标量指数 |
| 返回值 | `Tensor` | 逐元素幂运算结果 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × | - | - |

#### 2.2.2 Shape 支持
`self` 支持 1-4 维，输出 shape 与 `self` 一致；标量指数不涉及 shape 泛化。

### 2.3 特殊限制说明
> 标量指数按 PyTorch 类型提升规则参与运算；`bool` 不支持，NPU 复数类型不纳入本重载考察。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_pow_Tensor_Scalar(self, shape, dtype):
        def op_calc(base, exponent):
            return torch.ops.aten.pow.Tensor_Scalar(base, exponent)

        def op_calc_1(base, exponent):
            y = torch.ops.aten.pow.Tensor_Scalar(base, exponent)
            return y + y

        base = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, base, 2)
```