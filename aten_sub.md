# aten.sub

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sub(
    Tensor self,
    Tensor other,
    *,
    Scalar alpha=1
) -> Tensor
```

此处对应 `aten.sub.Tensor` 张量重载。

### 1.2 算子功能

`aten.sub` 对两输入逐元素做缩放减法：`out[i] = self[i] - alpha * other[i]`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 被减数张量 |
| `other` | `Tensor` | 减数张量，形状需与 self 可广播 |
| `alpha` | `Scalar` | 缩放系数，默认为 1，仅限关键字传入 |
| 返回值 | `Tensor` | 广播后形状的差值张量 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、shape 维度范围：[1-4]。

2、当前测试要求两输入同 shape。

### 2.3 特殊限制说明

> 不支持 bool 输入；整数结果要求 alpha 为整数。当前测试两输入同 dtype、同设备，使用默认 `alpha=1`。

### 2.4 使用方法

`_generate_tensor(shape, dtype)` 用于生成对应设备的测试张量。

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_sub(self, shape, dtype):
        def op_calc(a, b):
            y = torch.ops.aten.sub(a, b)
            return y

        def op_calc_1(a, b):
            y = torch.ops.aten.sub(a, b)
            return y + y

        a = self._generate_tensor(shape, dtype)
        b = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, a, b)
```
