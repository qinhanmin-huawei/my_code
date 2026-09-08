# aten.maximum.out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.maximum.out(
    Tensor self,
    Tensor other,
    *,
    Tensor(a!) out
) -> Tensor(a!)
```

### 1.2 算子功能

`aten.maximum.out` 对两输入逐元素取最大值并写入 `out`：`out[i] = max(self[i], other[i])`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 第一个输入张量 |
| `other` | `Tensor` | 第二个输入张量，形状须与 `self` 可广播 |
| `out` | `Tensor(a!)` | 写回张量；返回值与其别名 |
| 返回值 | `Tensor(a!)` | 写入结果后的 `out` |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、目标 shape 维度范围：[1-4]。

2、当前测试要求两输入同 shape。`out.shape == self.shape`。

### 2.3 特殊限制说明

> 当前测试要求 `self`、`other`、`out` 同 dtype、同 shape、同设备，out 独立分配，仅测无梯度前向。不支持复数输入；任一对应元素为 NaN 时结果为 NaN。

### 2.4 使用方法

`_generate_tensor(shape, dtype)` 用于生成对应设备的测试张量。

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_maximum_out(self, shape, dtype):
        def op_calc(a, b, out):
            y = torch.ops.aten.maximum.out(a, b, out=out)
            return y

        def op_calc_1(a, b, out):
            y = torch.ops.aten.maximum.out(a, b, out=out)
            return y + y

        a = self._generate_tensor(shape, dtype)
        b = self._generate_tensor(shape, dtype)
        out = torch.empty_like(a)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, a, b, out)
```
