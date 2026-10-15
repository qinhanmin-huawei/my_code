# aten.slice.t

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.slice.t(
    t[] l,
    int? start=None,
    int? end=None,
    int step=1
) -> t[]
```

### 1.2 算子功能

`aten.slice.t` 截取泛型列表，返回包含所选元素的新列表：`l[start:end:step]`，不复制元素对象。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `l` | `t[]` | 元素类型一致的列表，t 为泛型元素类型 |
| `start` | `int?` | 起始位置，包含；None 使用对应步长方向的默认起点 |
| `end` | `int?` | 结束位置，不包含；None 使用对应步长方向的默认终点 |
| `step` | `int` | 非零整数步长，默认 1，负值表示反向切片 |
| 返回值 | `t[]` | 元素类型不变的新列表 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

输入为泛型列表，元素类型由 `t` 决定，不适用 GPU/NPU Tensor dtype 表；下方 `List[int]` 只是使用示例，不将泛型规格限制为整数。

#### 2.2.2 Shape 支持

不适用 Tensor shape；支持不同长度的列表、空列表和空切片。

### 2.3 特殊限制说明

> `step` 不能为 0，支持负索引和越界裁剪。当前环境下，直接向 `torch.ops.aten.slice.t` 传 Python 列表会触发泛型 `t` 的绑定错误，示例采用类型化 TorchScript 调用。

### 2.4 使用方法

以下函数中的列表切片对应 `aten::slice.t`。

```python
import torch

unit = torch.jit.CompilationUnit("""
def list_slice(a: List[int], start: Optional[int], end: Optional[int], step: int) -> List[int]:
    return a[start:end:step]
""")

y = unit.list_slice([0, 1, 2, 3, 4, 5], 1, 5, 2)
assert y == [1, 3]
```
