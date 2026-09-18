# aten.sum.DimnameList_out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sum.DimnameList_out(
    Tensor self,
    Dimname[1] dim,
    bool keepdim=False,
    *,
    ScalarType? dtype=None,
    Tensor(a!) out
) -> Tensor(a!)
```

该原型对应仍保留 Named Tensor 的旧版接口。

### 1.2 算子功能

沿指定名称的维度求和，结果写入 `out`；`keepdim=True` 时保留被归约维度为 1。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 带维度名称的输入张量 |
| `dim` | `Dimname[1]` | 待归约的维度名称，Python 中传字符串列表 |
| `keepdim` | `bool` | 是否保留被归约维度，默认 False |
| `dtype` | `ScalarType?` | 可选的归约结果类型 |
| `out` | `Tensor(a!)` | 保存归约结果的独立输出张量 |
| 返回值 | `Tensor(a!)` | 写入结果的 out，与传入 out 别名 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self、out | - | - | - | - | - | - | - | - | - | - | - | - |
| NPU | self、out | - | - | - | - | - | - | - | - | - | - | - | - |

该算子是社区扩充的 dim name 特性，该特性不成熟，无法构造用例。

#### 2.2.2 Shape 支持

暂未验证 Shape 支持范围。

### 2.3 特殊限制说明

> 输入须携带与 `dim` 匹配的维度名称，归约维度不能重复。当前规格要求 `out` 的 shape、dtype 和设备与归约结果匹配。

### 2.4 使用方法

暂无可用测试用例。
