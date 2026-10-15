# aten.sin.out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能

`aten.sin.out` 按弧度逐元素计算正弦并写入 `out`：`out[i] = sin(self[i])`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `out` | `Tensor(a!)` | 预分配的写回张量 |
| 返回值 | `Tensor(a!)` | 写入结果后的 out，与 out 别名 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

当前 `out` dtype：浮点输入与输入一致；整数输入使用 `torch.get_default_dtype()`（通常为 fp32）。

| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、shape 维度范围：[1-4]。

2、当前测试要求 `out.shape == self.shape`。

### 2.3 特殊限制说明

> bool 不纳入当前支持规格；当前 out 与输入同设备，独立分配，仅测无梯度前向。实数 ±Inf 输入返回 NaN。

### 2.4 使用方法

`_generate_tensor(shape, dtype)` 用于生成对应设备的测试张量。

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_sin_out(self, shape, dtype):
        def op_calc(tensor, out):
            y = torch.ops.aten.sin.out(tensor, out=out)
            return y

        def op_calc_1(tensor, out):
            y = torch.ops.aten.sin.out(tensor, out=out)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        out_dtype = tensor.dtype if tensor.is_floating_point() else torch.get_default_dtype()
        out = torch.empty_like(tensor, dtype=out_dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, out)
```
