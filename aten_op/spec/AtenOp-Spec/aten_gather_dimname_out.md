# aten.gather.dimname_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.gather.dimname_out(
    Tensor self,
    Dimname dim,
    Tensor index,
    *,
    bool sparse_grad=False,
    Tensor(a!) out
) -> Tensor(a!)
```

### 1.2 算子功能
按照命名维度 `dim` 和 `index` 指定的位置，从 `self` 中收集元素，结果写入 `out` 并返回 `out`，是 `aten.gather.dimname` 的 out 变体。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 携带 named tensor 维度名称的输入 Tensor，提供数据来源 |
| `dim` | `Dimname` | 执行 gather 的命名维度，必须存在于 `self.names` 中 |
| `index` | `Tensor` | 收集位置索引，dtype 必须为 `int64` |
| `sparse_grad` | `bool` | 仅前向使用；为 True 时反向使用稀疏梯度，默认 False |
| `out` | `Tensor(a!)` | 用于承接收集结果的输出 Tensor |
| 返回值 | `Tensor(a!)` | 收集结果，即 `out` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| NPU | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。

#### 2.2.2 Shape 支持
输出 shape 与 `index.shape` 相同；`out` 的 shape 须与 `index` 一致，dtype 须与 `self` 兼容。

### 2.3 特殊限制说明
> `self` 必须携带与 `dim` 匹配的维度名称，且 names 不能重复。`index.dtype` 必须为 `torch.int64`，`out` 的 shape 和 dtype 须与收集结果匹配。

## 3. 遗留问题
该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。
