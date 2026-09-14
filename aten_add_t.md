# aten.add.t

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.add.t(t[] a, t[] b) -> t[]
```

### 1.2 算子功能
`aten.add.t` 对两个同类型的 Python 列表执行拼接，按顺序返回 `a` 的所有元素和 `b` 的所有元素。该重载不执行 Tensor 逐元素加法，Tensor 加法应使用 `aten.add.Tensor`。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `t[]` | 第一个待拼接列表 |
| `b` | `t[]` | 第二个待拼接列表 |
| 返回值 | `t[]` | 按 `a` 、`b` 顺序拼接后的新列表 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
该重载的输入是泛型列表，不是 Tensor 输入，因此不适用 Tensor DataType 支持表。两个列表的元素类型必须一致。

#### 2.2.2 Shape 支持
1、`a` 和 `b` 为一维 Python 列表，列表长度可不同。

2、返回列表长度为 `len(a) + len(b)`，元素顺序与输入列表保持一致。

### 2.3 特殊限制说明
> 该重载实际为列表拼接算子，不属于 Tensor 算子，无需生成 Triton kernel。由于现有 Inductor 测试框架要求 Tensor 结果以校验 dtype，不对该重载进行 Inductor lowering 测试。直接传入未标注的 Python `list` 可能无法让泛型参数 `t` 解析。

### 2.4 使用方法
