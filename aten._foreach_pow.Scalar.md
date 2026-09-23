# aten._foreach_pow.Scalar

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._foreach_pow.Scalar(self: Tensor[], exponent: Scalar) -> Tensor[]
```

### 1.2 算子功能

将输入列表中的每个张量提升到同一个标量指数。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor[]` | 底数张量列表。 |
| `exponent` | `Scalar` | 应用于每个输入张量的指数。 |
| `返回值` | `Tensor[]` | 与输入列表等长的结果列表。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |

#### 2.2.2 Shape 支持

1. `shape` 遍历值：`(4,)`、`(4, 8)`、`(4, 8, 16)`、`(4, 8, 16, 32)`。
2. 输入列表长度固定为 `2`，每个元素使用相同的遍历 shape；`exponent` 固定为 `2`；底数范围固定为 `[1, 4)`。

### 2.3 特殊限制说明

指数必须与输入 dtype 形成有效的幂运算。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__foreach_pow_scalar(self, shape, dtype):
        def op_calc(tensors):
            return torch.ops.aten._foreach_pow.Scalar(tensors, 2)

        def op_calc_1(tensors):
            y = torch.ops.aten._foreach_pow.Scalar(tensors, 2)
            return [item + item for item in y]

        tensors = [self._generate_tensor(shape, dtype, min=1, max=4) for _ in range(2)]
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensors)
```
