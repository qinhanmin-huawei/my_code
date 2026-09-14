# aten.mean.dtype_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.mean.dtype_out(Tensor self, *, ScalarType? dtype, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能
对输入 Tensor 的全部元素求平均值，并按可选 `dtype` 计算或转换结果后写入 `out` Tensor。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入 Tensor |
| `dtype` | `ScalarType?` | 可选计算/输出 dtype |
| `out` | `Tensor(a!)` | 保存零维结果的输出 Tensor |
| 返回值 | `Tensor(a!)` | 返回 `out` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | √ | √ |
| NPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | - | - |

#### 2.2.2 Shape 支持
输入支持 1-4 维；`out` 必须为零维 Tensor，dtype 与计算结果兼容。

### 2.3 特殊限制说明
> `out` 为零维输出；输入应为浮点 Tensor，NPU 复数类型不纳入本重载考察。

### 2.4 使用方法
```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_mean_dtype_out(self, shape, dtype):
        def op_calc(tensor, out_tensor):
            return torch.ops.aten.mean.dtype_out(tensor, dtype=torch.float32,
                                                  out=out_tensor)

        def op_calc_1(tensor, out_tensor):
            y = torch.ops.aten.mean.dtype_out(tensor, dtype=torch.float32,
                                               out=out_tensor)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        out_tensor = torch.empty((), dtype=torch.float32, device='npu')
        self.execute_inductor_test((), 'float32', op_calc, op_calc_1,
                                   tensor, out_tensor)
```