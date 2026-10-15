# aten.new_empty.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.new_empty.default(
    Tensor self,
    SymInt[] size,
    *,
    ScalarType? dtype=None,
    Layout? layout=None,
    Device? device=None,
    bool? pin_memory=None
) -> Tensor
```

### 1.2 算子功能
创建一个 shape 由 `size` 指定、元素未初始化的新 Tensor。未显式指定 `dtype`、`layout` 或 `device` 时，对应属性继承自 `self`；`self` 的数据内容不参与计算。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 提供默认 dtype、layout 和 device 的参考 Tensor，其数据内容不被读取 |
| `size` | `SymInt[]` | 新 Tensor 的 shape |
| `dtype` | `ScalarType?` | 可选输出 dtype；默认 `None`，即继承 `self.dtype` |
| `layout` | `Layout?` | 可选输出布局；默认 `None`，即继承 `self.layout` |
| `device` | `Device?` | 可选输出设备；默认 `None`，即继承 `self.device` |
| `pin_memory` | `bool?` | 是否使用 pinned memory；默认 `None` |
| 返回值 | `Tensor` | 指定 shape 和属性的未初始化 Tensor |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
本规格覆盖 1-4 维 `self` 和 1-4 维输出 Tensor。输出 shape 由 `size` 决定，不要求与 `self.shape` 相同。

### 2.3 特殊限制说明
> 返回 Tensor 的内存未初始化，不能对其初始元素值作任何假设，也不能直接比较两次调用得到的数值。默认情况下输出继承 `self` 的 dtype、layout 和 device；显式属性必须是目标后端支持的组合。本规格仅考察 strided layout，NPU pinned memory 不纳入考察，复数类型不纳入本重载考察。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_new_empty_default(self, shape, dtype):
        def op_calc(tensor, size):
            result = torch.ops.aten.new_empty.default(tensor, size)
            return result.zero_()

        def op_calc_1(tensor, size):
            result = torch.ops.aten.new_empty.default(tensor, size)
            return result.zero_()

        tensor = self._generate_tensor(shape, dtype)
        size = tuple(shape)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tensor, size)
```