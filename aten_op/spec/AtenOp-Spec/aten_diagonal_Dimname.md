# aten.diagonal.Dimname

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.diagonal.Dimname(
    Tensor(a) self,
    *,
    Dimname outdim,
    Dimname dim1,
    Dimname dim2,
    int offset=0
) -> Tensor(a)
```

### 1.2 算子功能
返回输入 Tensor 中由两个命名维度 `dim1`、`dim2` 确定的对角线元素组成的 view，对角维度被移除并以名称 `outdim` 追加在输出最后，`offset` 控制对角线相对主对角线的偏移，是 `aten.diagonal` 的 named tensor 重载。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor(a)` | 携带 named tensor 维度名称的输入 Tensor |
| `outdim` | `Dimname` | 输出中对角维度的名称 |
| `dim1` | `Dimname` | 对角线的第一个命名维度，必须存在于 `self.names` 中 |
| `dim2` | `Dimname` | 对角线的第二个命名维度，必须存在于 `self.names` 中 |
| `offset` | `int` | 对角线相对主对角线的偏移量，默认 0 |
| 返回值 | `Tensor(a)` | 对角线元素构成的 view |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| NPU | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。

#### 2.2.2 Shape 支持
输出的 shape 为移除 `dim1`、`dim2` 两个维度后，在末尾追加长度为对角线长度（由两维度长度和 `offset` 共同决定）的 `outdim` 维度；其余维度 shape 和名称保持不变。

### 2.3 特殊限制说明
> `self` 必须携带 named tensor 维度名称，且 names 不能重复；`dim1` 与 `dim2` 不能相同，返回的是 view，不修改 `self`。

## 3. 遗留问题
该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。
