# aten.prod.dim_Dimname

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.prod.dim_Dimname(
    Tensor self,
    Dimname dim,
    bool keepdim=False,
    *,
    ScalarType? dtype=None
) -> Tensor
```

### 1.2 算子功能
沿输入 Tensor 的指定命名维度执行逐元素乘积归约，返回归约结果。`keepdim=False` 时移除被归约的命名维度；`keepdim=True` 时保留该维度并将其长度置为 1。可通过 `dtype` 指定归约计算和输出类型。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 必须携带 named tensor 维度名称的输入 Tensor |
| `dim` | `Dimname` | 待归约的单个命名维度，必须存在于 `self` 的 names 中 |
| `keepdim` | `bool` | 是否保留被归约维度；默认为 `False` |
| `dtype` | `ScalarType?` | 可选的归约计算和输出 dtype；默认为 `None`，按算子类型推导 |
| 返回值 | `Tensor` | 沿指定命名维度归约后的 Tensor |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | × | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

`dtype` 显式指定时，实际支持范围还受目标设备归约 kernel 和类型转换能力限制；复数类型不纳入 NPU 本重载考察。

#### 2.2.2 Shape 支持
输入支持 1-4 维 Tensor。若输入 shape 为 `(s0, ..., sk, ..., sn)`，归约命名维度对应下标 `k`：

- `keepdim=False`：输出移除第 `k` 个维度，shape 为 `(s0, ..., s{k-1}, s{k+1}, ..., sn)`；
- `keepdim=True`：输出保留第 `k` 个维度，长度变为 1，shape 为 `(s0, ..., 1, ..., sn)`。

输出 names 按 named tensor 归约规则同步更新。

### 2.3 特殊限制说明
> `self` 必须携带与 `dim` 匹配的 named tensor 维度，`dim` 只能指定一个存在的命名维度。输入 Tensor 的 names 不能包含重复名称；空 Tensor、溢出行为以及显式 `dtype` 的累积精度由具体后端实现决定。

## 3. 遗留问题
该算子是社区扩充的 Dim name 特性，该特性不成熟，无法构造用例。