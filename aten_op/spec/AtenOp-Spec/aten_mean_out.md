# aten.mean.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.mean.out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能
沿可选的整数维度 `dim` 对输入 Tensor 求平均值，并将结果写入 `out` Tensor。当 `dim=None` 时对所有维度求平均值，返回零维 Tensor；`keepdim=True` 时保留被规约维度。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入 Tensor |
| `dim` | `int[]?` | 待规约的维度；`None` 表示全部维度 |
| `keepdim` | `bool` | 是否保留被规约维度，默认为 `False` |
| `dtype` | `ScalarType?` | 可选计算和输出 dtype |
| `out` | `Tensor(a!)` | 保存平均值结果的输出 Tensor |
| 返回值 | `Tensor(a!)` | 返回 `out` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | √ | √ |
| NPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | - | - |

#### 2.2.2 Shape 支持
输入支持 1-4 维；当 `dim=None` 且 `keepdim=False` 时，`out` 必须为零维 Tensor；指定维度时，`out` shape 应与对应的规约结果一致。

### 2.3 特殊限制说明
> `out` 的设备和 dtype 应与结果兼容；输入应为浮点 Tensor，NPU 复数类型不纳入本重载考察。

### 2.4 使用方法
```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_mean_out(self, shape, dtype):
        def op_calc(tensor, out_tensor):
            return torch.ops.aten.mean.out(tensor, None, False,
                                            out=out_tensor)

        def op_calc_1(tensor, out_tensor):
            y = torch.ops.aten.mean.out(tensor, None, False,
                                         out=out_tensor)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        out_tensor = torch.empty((), dtype=eval('torch.' + dtype), device='npu')
        self.execute_inductor_test((), dtype, op_calc, op_calc_1,
                                   tensor, out_tensor)
```