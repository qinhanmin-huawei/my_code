# aten.__and__.Tensor

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.__and__.Tensor(self: Tensor, other: Tensor) -> Tensor
```

### 1.2 算子功能

执行按位与运算。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 第一个输入张量。 |
| `other` | `Tensor` | 第二个输入张量。 |
| `返回值` | `Tensor` | 逐元素按位与结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | - | - | - | √ |
| NPU | - | √ | - | - | - | √ | - | √ | - | - | - | √ |

#### 2.2.2 Shape 支持

`shape` 遍历 `(4,)`、`(4, 8)`、`(4, 8, 16)`、`(4, 8, 16, 32)`，两个 Tensor 输入形状相同。

### 2.3 特殊限制说明

两个输入使用相同 dtype；不覆盖无符号整数和 int16。

### 2.4 使用方法

```python
result = torch.ops.aten.__and__.Tensor(self, other)
```
