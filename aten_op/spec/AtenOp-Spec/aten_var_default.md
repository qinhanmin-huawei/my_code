# aten.var.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.var.default(Tensor self, bool unbiased=True) -> Tensor
```

### 1.2 算子功能

对全部元素计算方差：`var = sum((x - mean(x))**2) / max(0, N - c)`。N 为每组归约元素数，c 在 unbiased=True 时为 1，否则为 0。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `unbiased` | `bool` | True 时除以 N−1，False 时除以 N；默认 True |
| 返回值 | `Tensor` | 方差，当前浮点规格下与输入同 dtype |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self | - | × | - | - | - | × | - | × | √ | √ | √ | × |
| NPU | self | - | × | - | - | - | × | - | × | √ | √ | √ | × |

int8、int32、int64 和 bool 输入社区不支持，当前 CPU Eager 和 NPU Eager 均不支持；无可用 Eager 精度参考，未测试对应 Inductor 路径。

#### 2.2.2 Shape 支持

1、`self` 的维度范围：[1-4]。

2、对全部元素归约，输出为零维张量。

### 2.3 特殊限制说明

> 空归约的方差为 NaN；N=1 且输入为有限值时，方差在 `unbiased=True` 下为 NaN，在 False 下为 0。

### 2.4 使用方法

```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    unbiaseds = [False, True]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    @parametrize('unbiased', unbiaseds)
    def test_aten_var_default(self, shape, dtype, unbiased):
        def op_calc(tensor, unbiased):
            y = torch.ops.aten.var.default(tensor, unbiased)
            return y

        def op_calc_1(tensor, unbiased):
            y = torch.ops.aten.var.default(tensor, unbiased)
            return y + y

        tensor = self._generate_tensor(shape, dtype, min=-4, max=4)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, unbiased)
```
