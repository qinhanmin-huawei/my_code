# aten.pow.Tensor_Scalar_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能
对输入张量执行标量指数幂运算，并将结果写入 `out`。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 幂运算底数张量 |
| `exponent` | `Scalar` | 标量指数 |
| `out` | `Tensor(a!)` | 输出张量，会被修改 |
| 返回值 | `Tensor(a!)` | 返回 `out` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × | - | - |

#### 2.2.2 Shape 支持
`self` 和 `out` 支持 1-4 维，`out.shape == self.shape`。

### 2.3 特殊限制说明
> `out` 的设备和 dtype 必须与结果兼容；`bool` 不支持，NPU 复数类型不纳入本重载考察。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_pow_Tensor_Scalar_out(self, shape, dtype):
        def op_calc(base, out_t):
            return torch.ops.aten.pow.Tensor_Scalar_out(base, 2, out=out_t)

        def op_calc_1(base, out_t):
            y = torch.ops.aten.pow.Tensor_Scalar_out(base, 2, out=out_t)
            return y + y

        base = self._generate_tensor(shape, dtype)
        out_t = torch.empty_like(base)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, base, out_t)
```