# aten.ne.Tensor

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.ne.Tensor(Tensor self, Tensor other) -> Tensor
```

### 1.2 算子功能
对 `self` 和 `other` 按 PyTorch 广播规则执行逐元素不等比较，返回 bool Tensor。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 第一个输入 Tensor |
| `other` | `Tensor` | 第二个输入 Tensor |
| 返回值 | `Tensor` | bool Tensor，shape 为广播后的 shape |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

返回 Tensor 的 dtype 固定为 `bool`。

#### 2.2.2 Shape 支持
输入支持 1-4 维，输出 shape 遵循 PyTorch 广播规则。

### 2.3 特殊限制说明
> 两个 Tensor 应处于同一设备，元素类型应兼容；复数输入不支持 NPU `complex128`。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_ne_Tensor(self, shape, dtype):
        def op_calc(first, second):
            return torch.ops.aten.ne.Tensor(first, second)

        def op_calc_1(first, second):
            y = torch.ops.aten.ne.Tensor(first, second)
            return y + y

        first = self._generate_tensor(shape, dtype)
        second = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   first, second)
```