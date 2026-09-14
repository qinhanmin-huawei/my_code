# aten.mean

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
```

### 1.2 算子功能
对输入 Tensor 的全部元素求算术平均值，返回零维 Tensor。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入 Tensor |
| `dtype` | `ScalarType?` | 可选的计算和输出 dtype |
| 返回值 | `Tensor` | 所有元素的平均值，结果为零维 Tensor |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | √ | √ |
| NPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | - | - |

#### 2.2.2 Shape 支持
输入支持 1-4 维，所有维度归约后输出零维 Tensor。

### 2.3 特殊限制说明
> `mean` 主要用于浮点或复数 Tensor；整型输入需先转换为支持的浮点 dtype。NPU 复数类型不纳入本重载考察。

### 2.4 使用方法
```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_mean(self, shape, dtype):
        def op_calc(tensor):
            return torch.ops.aten.mean(tensor)

        def op_calc_1(tensor):
            y = torch.ops.aten.mean(tensor)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test((), dtype, op_calc, op_calc_1, tensor)
```