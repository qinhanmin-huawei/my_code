# aten.select.Dimname

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.select.Dimname(Tensor(a) self, str dim, int index) -> Tensor(a)
```

### 1.2 算子功能
沿输入 Tensor 的命名维度 `dim` 选择下标为 `index` 的单个切片并返回 view，同时移除被选择的命名维度。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 携带 named tensor 维度名称的输入 Tensor |
| `dim` | `str` | 要选择的命名维度，必须存在于 `self.names` |
| `index` | `int` | 被选择的切片下标 |
| 返回值 | `Tensor` | 选择后的 Tensor view |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
输出删除命名维度 `dim` 对应的维度，其他维度 shape 和 names 保持不变。

### 2.3 特殊限制说明
> `self` 必须携带与 `dim` 匹配的维度名称，且 names 不能重复；`index` 必须位于对应维度的合法范围内。该算子返回 view，不修改 `self`。

## 3. 遗留问题
该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。