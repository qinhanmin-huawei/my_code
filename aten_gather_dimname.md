# aten.gather.dimname

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.gather.dimname(
    Tensor self,
    Dimname dim,
    Tensor index,
    *,
    bool sparse_grad=False
) -> Tensor
```

### 1.2 算子功能
按照命名维度 `dim` 和 `index` 指定的位置，从 `self` 中收集元素并返回新 Tensor，是 `aten.gather` 的 named tensor 重载。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 携带 named tensor 维度名称的输入 Tensor，提供数据来源 |
| `dim` | `Dimname` | 执行 gather 的命名维度，必须存在于 `self.names` 中 |
| `index` | `Tensor` | 收集位置索引，dtype 必须为 `int64` |
| `sparse_grad` | `bool` | 仅前向使用；为 True 时反向使用稀疏梯度，默认 False |
| 返回值 | `Tensor` | 收集结果，shape 与 `index` 相同 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| NPU | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。

#### 2.2.2 Shape 支持
输出 shape 与 `index.shape` 相同；`index` 的维数必须与 `self` 相同，除 `dim` 外其余维度长度不大于 `self` 对应维度，索引值必须位于 `dim` 维度的合法范围内。

### 2.3 特殊限制说明
> `self` 必须携带与 `dim` 匹配的维度名称，且 names 不能重复。`index.dtype` 必须为 `torch.int64`，`self` 与 `index` 的 device 应兼容。

## 3. 遗留问题
该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。
