# aten.sum.dim_DimnameList

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sum.dim_DimnameList(
    Tensor self,
    Dimname[1] dim,
    bool keepdim=False,
    *,
    ScalarType? dtype=None
) -> Tensor
```

该原型对应仍保留 Named Tensor 的旧版接口。

### 1.2 算子功能

沿指定名称的维度求和；`keepdim=True` 时保留被归约维度为 1。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 带维度名称的输入张量 |
| `dim` | `Dimname[1]` | 待归约的维度名称，Python 中传字符串列表 |
| `keepdim` | `bool` | 是否保留被归约维度，默认 False |
| `dtype` | `ScalarType?` | 可选的归约结果类型 |
| 返回值 | `Tensor` | 按指定名称维度求和后的张量 |

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

> 输入须携带与 `dim` 匹配的维度名称，归约维度不能重复。

### 2.4 使用方法

暂无可用测试用例。
