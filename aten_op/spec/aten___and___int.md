# aten.__and__.int

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.__and__.int(a: int, b: int) -> int
```

### 1.2 算子功能

执行按位与运算。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `int` | 第一个整数标量。 |
| `b` | `int` | 第二个整数标量。 |
| `返回值` | `int` | 整数按位与结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | - | - | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | - | - | - | - |

#### 2.2.2 Shape 支持

算子输入为标量，无 Tensor 维度；验证使用固定 1 维辅助 Tensor，shape 为 `(64,)`。

### 2.3 特殊限制说明

dtype 遍历 int8、int32、int64；标量值固定为 6 和 2，辅助 Tensor 固定为 int64、shape `(64,)`。

### 2.4 使用方法

```python
result = torch.ops.aten.__and__.int(6, 2)
```
