# aten.cat.names_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.cat.names_out(
    Tensor[] tensors,
    Dimname dim,
    *,
    Tensor(a!) out
) -> Tensor(a!)
```

### 1.2 算子功能
沿命名维度 `dim` 将 `tensors` 中的 Tensor 按顺序在给定维度上拼接，结果写入 `out` 并返回 `out`，是 `aten.cat.names` 的 out 变体。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `tensors` | `Tensor[]` | 待拼接的 Tensor 列表，各 Tensor 携带 named tensor 维度名称，列表不能为空 |
| `dim` | `Dimname` | 拼接维度的名称，Python 中传字符串 |
| `out` | `Tensor(a!)` | 用于承接拼接结果的输出 Tensor |
| 返回值 | `Tensor(a!)` | 拼接后的结果，即 `out` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| NPU | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。

#### 2.2.2 Shape 支持
输出在 `dim` 维度上的长度为各输入 Tensor 对应维度长度之和，其余维度 shape 与输入一致；`out` 的 shape 须与拼接结果一致。

### 2.3 特殊限制说明
> 所有输入 Tensor 必须携带 named tensor 维度名称，除 `dim` 外其余维度的名称必须一致且不能重复；各输入的 dtype 应相同，`out` 的 dtype 和 shape 须与拼接结果匹配。

## 3. 遗留问题
该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。
