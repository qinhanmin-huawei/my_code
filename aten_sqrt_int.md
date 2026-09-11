# aten.sqrt.int

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sqrt.int(int a) -> float
```

### 1.2 算子功能

`aten.sqrt.int` 计算整数标量 `a` 的平方根，返回 float。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `int` | 输入数值标量 |
| 返回值 | `float` | 平方根结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

下表按构造输入标量的来源 dtype 列示，实际参数为 Python 标量。

| 设备 | 参数 | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | `a` | √ | √ | √ | - | - | - | × | - | - |
| NPU | `a` | √ | √ | √ | - | - | - | × | - | - |

#### 2.2.2 Shape 支持

不适用，输入与返回值均为标量。

### 2.3 特殊限制说明

> `a` 须在 ATen int（有符号 64 位整数）范围内；负整数输入返回 NaN。
>
> 当前测试中的标量计算被常量折叠，不作为设备端算子精度或性能验证。

### 2.4 使用方法

```python
import torch

y = torch.ops.aten.sqrt.int(9)  # 3.0
```
