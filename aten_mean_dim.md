# aten.mean.dim

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
```

### 1.2 算子功能
沿指定整数维度求平均值；`keepdim=True` 时保留被归约维度为 1。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入 Tensor |
| `dim` | `int[1]?` | 待归约维度，可为多个维度 |
| `keepdim` | `bool` | 是否保留归约维度 |
| `dtype` | `ScalarType?` | 可选的计算和输出 dtype |
| 返回值 | `Tensor` | 按指定维度归约后的结果 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | √ | √ |
| NPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | - | - |

#### 2.2.2 Shape 支持
输入支持 1-4 维；输出 shape 按 `dim` 删除对应维度，或在 `keepdim=True` 时将其置为 1。

### 2.3 特殊限制说明
> `dim` 使用整数维度索引；输入应为浮点 Tensor。NPU 复数类型不纳入本重载考察。

### 2.4 使用方法
```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_mean_dim(self, shape, dtype):
        def op_calc(tensor):
            return torch.ops.aten.mean.dim(tensor, [0], False)

        def op_calc_1(tensor):
            y = torch.ops.aten.mean.dim(tensor, [0], False)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape[1:] if len(shape) > 1 else (), dtype,
                                   op_calc, op_calc_1, tensor)
```