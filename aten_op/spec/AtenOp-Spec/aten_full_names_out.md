# aten.full.names_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.full.names_out(
    int[] size,
    Scalar fill_value,
    *,
    Dimname[]? names,
    Tensor(a!) out
) -> Tensor(a!)
```

### 1.2 算子功能
以 `fill_value` 填充输出 Tensor `out` 并返回 `out`，`out` 的 shape 由 `size` 决定、维度名称由 `names` 决定，是 `aten.full.names` 的 out 变体（由 `full.names` autogen 生成）。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `size` | `int[]` | 输出 Tensor 各维度的长度 |
| `fill_value` | `Scalar` | 用于填充输出 Tensor 的标量值 |
| `names` | `Dimname[]?` | 输出 Tensor 的维度名称列表，可选，Python 中传字符串列表 |
| `out` | `Tensor(a!)` | 用于承接填充结果的输出 Tensor |
| 返回值 | `Tensor(a!)` | 填充后的结果，即 `out` |

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
> `names` 中不能出现重复的维度名称；`out` 需预先创建，其 shape 须与 `size` 一致。

## 3. 遗留问题
该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。
