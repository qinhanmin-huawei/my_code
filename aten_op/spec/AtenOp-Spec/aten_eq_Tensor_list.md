# aten.eq.Tensor_list

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.eq.Tensor_list(
    Tensor[] a,
    Tensor[] b
) -> bool
```

### 1.2 算子功能
比较两个 Tensor 列表是否相等，返回单个 Python `bool`，是 `aten.ne.Tensor_list` 对应的相等比较重载。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `Tensor[]` | 第一个 Tensor 列表 |
| `b` | `Tensor[]` | 第二个 Tensor 列表 |
| 返回值 | `bool` | 两个列表相等时为 `True` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| NPU | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

该 `*_list` overload 与 TorchScript 的列表类型系统有关，无法构造用例，未进行 dtype 验证。返回值是 Python `bool`，不是 bool Tensor。

#### 2.2.2 Shape 支持
输入为 Tensor 列表，不涉及固定 shape 约束。

### 2.3 特殊限制说明
> `Tensor[]` / `*_list` overload 主要与 TorchScript 的列表类型系统有关，而 TorchScript 架构已被社区弃用。该 overload 目前仍保留在 schema 中，但不代表仍会持续完善。

## 3. 遗留问题
该算子与 TorchScript 的列表类型系统有关，TorchScript 架构已被社区弃用，无法构造用例。
