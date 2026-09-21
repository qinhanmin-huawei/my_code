# aten._foreach_sub.Scalar

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._foreach_sub.Scalar(self: Tensor[], scalar: Scalar) -> Tensor[]
```

### 1.2 算子功能

从输入列表的每个张量中减去同一个标量。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor[]` | 输入张量列表。 |
| `scalar` | `Scalar` | 从每个张量中减去的固定标量。 |
| `返回值` | `Tensor[]` | 与输入列表等长的结果列表。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |


#### 2.2.2 Shape 支持

1. `shape` 遍历值：`(4,)`、`(4, 8)`、`(4, 8, 16)`、`(4, 8, 16, 32)`。
2. 输入列表长度固定为 `2`，每个元素使用相同的遍历 shape；`scalar` 固定为 `2`；输入值范围固定为 `[-8, 8)`。

### 2.3 特殊限制说明

两个 Tensor 列表中的对应元素必须具有相同 shape；不覆盖复数和 bool。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__foreach_sub_scalar(self, shape, dtype):
        def op_calc(tensors):
            return torch.ops.aten._foreach_sub.Scalar(tensors, 2)

        def op_calc_1(tensors):
            result = torch.ops.aten._foreach_sub.Scalar(tensors, 2)
            return [item + item for item in result]

        tensors = [self._generate_tensor(shape, dtype, min=-8, max=8) for _ in range(2)]
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensors)
```
