# aten.full.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.full.default(
    SymInt[] size,
    Scalar fill_value,
    *,
    ScalarType? dtype=None,
    Layout? layout=None,
    Device? device=None,
    bool? pin_memory=None
) -> Tensor
```

### 1.2 算子功能
创建 shape 由 `size` 指定、所有元素均初始化为 `fill_value` 的 Tensor。可通过 `dtype`、`layout`、`device` 和 `pin_memory` 指定输出属性；未指定时按 PyTorch 默认规则推导。该算子是 `aten.full` 的默认 overload。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `size` | `SymInt[]` | 输出 Tensor 的 shape |
| `fill_value` | `Scalar` | 用于初始化输出各元素的标量值 |
| `dtype` | `ScalarType?` | 可选输出 dtype；默认为 `None`，按 PyTorch 默认类型规则推导 |
| `layout` | `Layout?` | 可选输出布局；默认为 `None`，使用默认布局 |
| `device` | `Device?` | 可选输出设备；默认为 `None`，使用默认设备 |
| `pin_memory` | `bool?` | 是否分配 pinned memory；默认为 `None` |
| 返回值 | `Tensor` | 指定 shape 和属性、元素均为 `fill_value` 的新 Tensor |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
本规格覆盖 1-4 维输出 Tensor，输出 shape 由 `size` 决定。

### 2.3 特殊限制说明
> `fill_value` 必须能转换为目标 `dtype`；超出目标 dtype 表示范围时可能报错或按 PyTorch 类型转换规则处理。用例显式指定 `dtype` 和目标设备，仅考察 strided layout；NPU pinned memory 和复数类型不纳入本规格考察。该算子创建并初始化新 Tensor，不读取外部输入数据。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_full_default(self, shape, dtype):
        def op_calc(size, value):
            return torch.ops.aten.full.default(
                size, value, dtype=eval('torch.' + dtype), device='npu')

        def op_calc_1(size, value):
            y = torch.ops.aten.full.default(
                size, value, dtype=eval('torch.' + dtype), device='npu')
            return y + y

        value = True if dtype == 'bool' else 1
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tuple(shape), value)
```