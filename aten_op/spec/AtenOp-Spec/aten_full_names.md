# aten.full.names

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.full.names(
    int[] size,
    Scalar fill_value,
    *,
    Dimname[]? names,
    ScalarType? dtype=None,
    Layout? layout=None,
    Device? device=None,
    bool? pin_memory=None
) -> Tensor
```

### 1.2 算子功能
按形状 `size` 创建一个以 `fill_value` 填充的新 Tensor，可选地通过 `names` 为各维度指定名称，是 `aten.full` 的 named tensor 工厂重载。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `size` | `int[]` | 输出 Tensor 各维度的长度 |
| `fill_value` | `Scalar` | 用于填充输出 Tensor 的标量值 |
| `names` | `Dimname[]?` | 输出 Tensor 的维度名称列表，可选，Python 中传字符串列表 |
| `dtype` | `ScalarType?` | 输出 Tensor 的 dtype，默认 None 时按 `fill_value` 推断 |
| `layout` | `Layout?` | 输出 Tensor 的 layout，默认 None 表示 strided |
| `device` | `Device?` | 输出 Tensor 所在设备，默认 None 表示当前设备 |
| `pin_memory` | `bool?` | 是否使用锁页内存，默认 None |
| 返回值 | `Tensor` | 以 `fill_value` 填充的新 Tensor |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| NPU | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。

#### 2.2.2 Shape 支持
输出 shape 由 `size` 决定；`names` 提供时其长度须与 `size` 一致，各名称一一对应且不能重复。

### 2.3 特殊限制说明
> `names` 中不能出现重复的维度名称；无 Tensor 输入，属于工厂函数。

## 3. 遗留问题
该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。
