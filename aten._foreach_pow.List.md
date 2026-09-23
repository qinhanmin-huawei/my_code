# aten._foreach_pow.List

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._foreach_pow.List(self: Tensor[], exponent: Tensor[]) -> Tensor[]
```

### 1.2 算子功能

对两个 Tensor 列表逐项执行幂运算，计算 `self[i] ** exponent[i]`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor[]` | 底数张量列表。 |
| `exponent` | `Tensor[]` | 指数张量列表。 |
| `返回值` | `Tensor[]` | 与输入列表等长的结果列表。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |
| NPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |

#### 2.2.2 Shape 支持

输入列表中的 Tensor 支持 1～4 维。

### 2.3 特殊限制说明

两个列表中的对应元素必须具有相同 shape 和 dtype；使用正底数避免非整数指数产生无效值，社区不支持int类型。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__foreach_pow_list(self, shape, dtype):
        def op_calc(tensors, exponent):
            return torch.ops.aten._foreach_pow.List(tensors, exponent)

        def op_calc_1(tensors, exponent):
            y = torch.ops.aten._foreach_pow.List(tensors, exponent)
            return [item + item for item in y]

        tensors = [self._generate_tensor(shape, dtype, min=1, max=4) for _ in range(2)]
        exponent = [self._generate_tensor(shape, dtype, min=1, max=4) for _ in range(2)]
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensors, exponent)
```
