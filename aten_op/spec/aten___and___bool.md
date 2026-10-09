# aten.__and__.bool

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.__and__.bool(a: bool, b: bool) -> bool
```

### 1.2 算子功能

执行按位与运算。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `bool` | 第一个布尔标量。 |
| `b` | `bool` | 第二个布尔标量。 |
| `返回值` | `bool` | 布尔按位与结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | - | - | - | - | - | - | - | - | √ |
| NPU | - | - | - | - | - | - | - | - | - | - | - | √ |

#### 2.2.2 Shape 支持

算子输入为标量，无 Tensor 维度；验证使用固定 1 维辅助 Tensor，shape 为 `(64,)`。

### 2.3 特殊限制说明

布尔标量固定为 True 和 True；辅助 Tensor 固定为 bool、shape `(64,)`，仅用于把标量结果接入融合计算。

### 2.4 使用方法

```python
result = torch.ops.aten.__and__.bool(True, True)
```
