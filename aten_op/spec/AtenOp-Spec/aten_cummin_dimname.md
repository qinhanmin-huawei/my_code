# aten.cummin.dimname

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.cummin.dimname(
    Tensor self,
    Dimname dim
) -> (Tensor values, Tensor indices)
```

### 1.2 算子功能
沿命名维度 `dim` 计算输入 Tensor 的累积最小值，返回累积最小值 `values` 及每个位置对应最小值首次出现位置的下标 `indices`，是 `aten.cummin` 的 named tensor 重载。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 携带 named tensor 维度名称的输入 Tensor |
| `dim` | `Dimname` | 执行累积最小值的命名维度，必须存在于 `self.names` 中 |
| 返回值 `values` | `Tensor` | 累积最小值，shape 与 `self` 相同 |
| 返回值 `indices` | `Tensor` | 累积最小值来源位置的下标，dtype 为 `int64`，shape 与 `self` 相同 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| NPU | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。

#### 2.2.2 Shape 支持
`values` 与 `indices` 的 shape 均与 `self.shape` 相同，维度名称保持不变。

### 2.3 特殊限制说明
> `self` 必须携带与 `dim` 匹配的维度名称，且 names 不能重复。累积最小值要求元素之间可比较，复数类型不适用。

## 3. 遗留问题
该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。
