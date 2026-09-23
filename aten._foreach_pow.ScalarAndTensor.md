# aten._foreach_pow.ScalarAndTensor

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._foreach_pow.ScalarAndTensor(self: Scalar, exponent: Tensor[]) -> Tensor[]
```

### 1.2 算子功能

以同一个标量为底数，对指数张量列表逐项执行幂运算。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Scalar` | 应用于每个指数张量的底数。 |
| `exponent` | `Tensor[]` | 指数张量列表。 |
| `返回值` | `Tensor[]` | 与指数列表等长的结果列表。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |
| NPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |

#### 2.2.2 Shape 支持

1. `shape` 遍历值：`(4,)`、`(4, 8)`、`(4, 8, 16)`、`(4, 8, 16, 32)`。
2. 指数列表长度固定为 `2`，每个元素使用相同的遍历 shape；底数固定为 `2`；指数范围固定为 `[1, 4)`。

### 2.3 特殊限制说明

标量底数与指数 dtype 必须形成有效的幂运算，社区不支持int类型。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__foreach_pow_scalar_and_tensor(self, shape, dtype):
        def op_calc(exponent):
            return torch.ops.aten._foreach_pow.ScalarAndTensor(2, exponent)

        def op_calc_1(exponent):
            y = torch.ops.aten._foreach_pow.ScalarAndTensor(2, exponent)
            return [item + item for item in y]

        exponent = [self._generate_tensor(shape, dtype, min=1, max=4) for _ in range(2)]
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, exponent)
```
