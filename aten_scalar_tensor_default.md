# aten.scalar_tensor.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.scalar_tensor.default(
    Scalar s,
    *,
    ScalarType? dtype=None,
    Layout? layout=None,
    Device? device=None,
    bool? pin_memory=None
) -> Tensor
```

### 1.2 算子功能
将 Python 标量 `s` 转换为零维 Tensor。可通过 `dtype`、`layout`、`device` 和 `pin_memory` 指定输出属性；未指定 `dtype` 时，按照输入标量类型和 PyTorch 默认类型规则推导。该重载是 `aten.scalar_tensor` 的默认 overload。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `s` | `Scalar` | 待转换的 Python 数值或布尔标量 |
| `dtype` | `ScalarType?` | 可选输出 dtype；默认按标量类型推导 |
| `layout` | `Layout?` | 可选输出布局；默认使用 strided layout |
| `device` | `Device?` | 可选输出设备；未指定时使用默认设备 |
| `pin_memory` | `bool?` | 是否分配 pinned memory；默认 `None` |
| 返回值 | `Tensor` | 包含 `s` 的零维 Tensor |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
输出固定为零维 Tensor，shape 为 `()`，仅包含一个元素。

### 2.3 特殊限制说明
> `s` 必须能转换为目标 `dtype`；超出目标 dtype 表示范围时，可能报错或按 PyTorch 类型转换规则处理。NPU 用例必须显式指定 `device='npu'`，否则算子通常在默认设备上创建 Tensor。本规格仅考察 strided layout，不考察 NPU pinned memory 和复数类型。Python 标量在编译时可能被特化，是否保留独立算子或生成设备 kernel 以实际 output code 为准。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']

    @parametrize('dtype', dtypes)
    def test_aten_scalar_tensor_default(self, dtype):
        def op_calc(value):
            return torch.ops.aten.scalar_tensor.default(
                value, dtype=eval('torch.' + dtype), device='npu')

        def op_calc_1(value):
            y = torch.ops.aten.scalar_tensor.default(
                value, dtype=eval('torch.' + dtype), device='npu')
            return y + y

        value = True if dtype == 'bool' else 1
        self.execute_inductor_test((), dtype, op_calc, op_calc_1, value)