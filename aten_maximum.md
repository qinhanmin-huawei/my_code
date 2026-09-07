# aten.maximum

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.maximum(Tensor self, Tensor other) -> Tensor
```

### 1.2 算子功能

`aten.maximum` 对两输入逐元素取最大值：`out[i] = max(self[i], other[i])`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 第一个输入张量 |
| `other` | `Tensor` | 第二个输入张量，形状须与 `self` 可广播 |
| 返回值 | `Tensor` | 广播后形状的结果；两输入同 dtype 时保持该 dtype |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、目标 shape 维度范围：[1-4]。

2、当前测试要求两输入同 shape。

### 2.3 特殊限制说明

> 当前测试要求两输入同 dtype、同设备；不支持复数输入。任一对应元素为 NaN 时结果为 NaN。

### 2.4 使用方法

`_generate_tensor(shape, dtype)` 用于生成对应设备的测试张量。

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_maximum(self, shape, dtype):
        def op_calc(a, b):
            y = torch.ops.aten.maximum(a, b)
            return y

        def op_calc_1(a, b):
            y = torch.ops.aten.maximum(a, b)
            return y + y

        a = self._generate_tensor(shape, dtype)
        b = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, a, b)
```
