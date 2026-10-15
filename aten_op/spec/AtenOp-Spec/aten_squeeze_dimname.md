# aten.squeeze.dimname

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.squeeze.dimname(Tensor(a) self, Dimname dim) -> Tensor(a)
```

该原型对应仍保留 Named Tensor 的旧版接口。

### 1.2 算子功能

删除名称为 `dim` 且长度为 1 的维度，返回与 `self` 共享存储的视图，元素值不变。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor(a)` | 带维度名称的输入张量 |
| `dim` | `Dimname` | 待压缩的维度名称，Python 中传字符串 |
| 返回值 | `Tensor(a)` | 保持 self 的 dtype 和设备的视图 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self | - | - | - | - | - | - | - | - | - | - | - | - |
| NPU | self | - | - | - | - | - | - | - | - | - | - | - | - |

该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。

#### 2.2.2 Shape 支持

暂未验证 Shape 支持范围。

### 2.3 特殊限制说明

> 输入须携带与 `dim` 匹配的维度名称。

### 2.4 使用方法

暂无可用测试用例。
