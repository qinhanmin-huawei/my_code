# aten.scatter.dimname_src

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.scatter.dimname_src(
    Tensor self,
    str dim,
    Tensor index,
    Tensor src
) -> Tensor
```

### 1.2 算子功能
按照命名维度 `dim` 和 `index` 指定的位置，将 `src` 中的元素写入 `self` 的副本并返回新 Tensor，不修改输入 `self`。该重载以维度名称代替整数维度下标。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 携带 named tensor 维度名称的输入 Tensor |
| `dim` | `str` | 执行 scatter 的命名维度，必须存在于 `self.names` 中 |
| `index` | `Tensor` | 写入位置索引，dtype 必须为 `int64` |
| `src` | `Tensor` | 待写入的数据 Tensor |
| 返回值 | `Tensor` | scatter 后的新 Tensor，shape 与 `self` 相同 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
输入为带名称的 Tensor，输出 shape 与 `self.shape` 相同。`index` 与 `src` 的维度数必须与 `self` 相同，且索引值必须位于命名维度对应的合法范围内。

### 2.3 特殊限制说明
> `self` 必须携带与 `dim` 匹配的维度名称，且 names 不能重复。`index.dtype` 必须为 `torch.int64`。重复索引可能产生非确定性；复数类型不纳入本规格考察。

## 3. 遗留问题
该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。