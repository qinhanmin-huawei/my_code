# aten.slice.str

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.slice.str(
    str string,
    int? start=None,
    int? end=None,
    int step=1
) -> str
```

### 1.2 算子功能

`aten.slice.str` 按起止位置和步长截取字符串，返回新字符串；对 ASCII 输入等价于 `string[start:end:step]`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `string` | `str` | 输入字符串，当前规格限定 ASCII |
| `start` | `int?` | 起始位置，包含；None 使用对应步长方向的默认起点 |
| `end` | `int?` | 结束位置，不包含；None 使用对应步长方向的默认终点 |
| `step` | `int` | 非零整数步长，默认 1，负值表示反向切片 |
| 返回值 | `str` | 切片后的字符串 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

输入和输出均为 `str`，不适用 GPU/NPU Tensor dtype 表。

#### 2.2.2 Shape 支持

不适用 Tensor shape；字符串长度可变，支持空字符串及空切片。

### 2.3 特殊限制说明

> `step` 不能为 0；支持负索引和越界裁剪。当前环境下，该入口在完整捕图时因返回 `str` 报 `torch.* op returned non-Tensor`；允许 graph break 时在图外执行。

### 2.4 使用方法

Eager 调用示例：

```python
import torch

y = torch.ops.aten.slice.str('abcdefgh', 1, 7, 2)
assert y == 'bdf'

y = torch.ops.aten.slice.str('abcdefgh', None, None, -1)
assert y == 'hgfedcba'
```
