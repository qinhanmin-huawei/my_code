# aten.var.names_dim

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.var.names_dim(
    Tensor self,
    Dimname[1] dim,
    bool unbiased=True,
    bool keepdim=False
) -> Tensor
```

该原型对应仍保留 Named Tensor 的接口。

### 1.2 算子功能

沿指定名称的维度计算方差：`var = sum((x - mean(x))**2) / max(0, N - c)`。N 为每组归约元素数，c 在 unbiased=True 时为 1，否则为 0。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 带维度名称的输入张量 |
| `dim` | `Dimname[1]` | 待归约的维度名称，Python 中传字符串列表 |
| `unbiased` | `bool` | True 时除以 N−1，False 时除以 N；默认 True |
| `keepdim` | `bool` | 是否将被归约维度保留为 1，默认 False |
| 返回值 | `Tensor` | 方差，当前浮点规格下与输入同 dtype |

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

> 输入须携带与 `dim` 匹配的维度名称，归约维度不能重复。当 N=1 且 `unbiased=True` 时，方差为 NaN。

### 2.4 使用方法

暂无可用测试用例。
