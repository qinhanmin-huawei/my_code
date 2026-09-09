# aten.pow

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.pow(Tensor self, Tensor exponent) -> Tensor
```

### 1.2 算子功能
对输入张量 `self` 和指数张量 `exponent` 执行逐元素幂运算，指数按广播规则参与计算。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 幂运算底数张量 |
| `exponent` | `Tensor` | 幂运算指数张量，可广播 |
| 返回值 | `Tensor` | 逐元素幂运算结果 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | √ | √ |
| NPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | - | - |
> 我们的代码和社区的代码都对int类型单独fallback
#### 2.2.2 Shape 支持
`self` 和 `exponent` 支持 1-4 维，形状遵循 PyTorch 广播规则，输出为广播后的形状。

### 2.3 特殊限制说明
> `complex128` 在 NPU 上不纳入本重载考察；`bool` 不支持。指数张量必须与底数张量的计算类型兼容。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_pow(self, shape, dtype):
        def op_calc(base, exponent):
            return torch.ops.aten.pow(base, exponent)

        def op_calc_1(base, exponent):
            y = torch.ops.aten.pow(base, exponent)
            return y + y

        base = self._generate_tensor(shape, dtype)
        exponent = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, base, exponent)

```