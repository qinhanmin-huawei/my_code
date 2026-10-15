# aten.eq.int_list

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.eq.int_list(
    int[] a,
    int[] b
) -> bool
```

### 1.2 算子功能
比较两个 Python 整数列表是否相等，返回 Python `bool`，是 `aten.ne.int_list` 对应的相等比较重载。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `int[]` | 第一个整数列表 |
| `b` | `int[]` | 第二个整数列表 |
| 返回值 | `bool` | 两个列表相等时为 `True` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| 实现域 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Host Scalar List | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

仅涉及 Python `int` 元素，不涉及 Tensor dtype；该 `*_list` overload 无法构造用例，未进行验证。

#### 2.2.2 Shape 支持
输入为一维 Python 整数列表，长度可以不同。

### 2.3 特殊限制说明
> 返回 Python `bool`，不涉及 GPU/NPU Tensor 计算。该 `*_list` overload 主要与 TorchScript 的列表类型系统有关，而 TorchScript 架构已被社区弃用。

## 3. 遗留问题
该算子与 TorchScript 的列表类型系统有关，TorchScript 架构已被社区弃用，无法构造用例。
