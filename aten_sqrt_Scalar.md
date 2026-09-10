# aten.sqrt.Scalar

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sqrt.Scalar(Scalar a) -> Scalar
```

### 1.2 算子功能

`aten.sqrt.Scalar` 计算标量 `a` 的平方根：`out = sqrt(a)`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `Scalar` | 输入数值标量 |
| 返回值 | `Scalar` | 平方根结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

下表按构造输入标量的来源 dtype 列示，实际参数为 Python 标量。

| 参数 | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `a` | √ | √ | √ | √ | √ | √ | × | - | - |

#### 2.2.2 Shape 支持

不适用，输入与返回值均为标量。

### 2.3 特殊限制说明

> 当前实数样例要求 `a >= 0`；负实数输入返回 NaN。
>
> 当前测试中的标量计算被常量折叠，不作为设备端算子精度或性能验证。

### 2.4 使用方法

```python
import torch

y = torch.ops.aten.sqrt.Scalar(4.0)  # 2.0
```
