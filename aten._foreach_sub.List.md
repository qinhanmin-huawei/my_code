# aten._foreach_sub.List

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._foreach_sub.List(self: Tensor[], other: Tensor[], *, alpha: Scalar=1) -> Tensor[]
```

### 1.2 算子功能

将两个 Tensor 列表逐项相减，计算 `self[i] - alpha * other[i]`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor[]` | 第一个输入张量列表。 |
| `other` | `Tensor[]` | 第二个输入张量列表。 |
| `alpha` | `Scalar` | 第二个列表的固定缩放系数。 |
| `返回值` | `Tensor[]` | 与输入列表等长的结果列表。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |

注：不支持int8和int64的npu的eager。
#### 2.2.2 Shape 支持

1. `shape` 遍历值：`(4,)`、`(4, 8)`、`(4, 8, 16)`、`(4, 8, 16, 32)`。
2. 两个输入列表长度均固定为 `2`，对应元素使用相同的遍历 shape；`alpha` 固定为 `2`；输入值范围固定为 `[-8, 8)`。

### 2.3 特殊限制说明

上游pytorch bug，编译时无法传递alpha，导致 alpha 参数丢失。链接：https://github.com/pytorch/pytorch/pull/184421。所以当alpha输入不为1时会精度错误。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__foreach_sub_list(self, shape, dtype):
        def op_calc(tensors, other):
            return torch.ops.aten._foreach_sub.List(tensors, other, alpha=2)

        def op_calc_1(tensors, other):
            result = torch.ops.aten._foreach_sub.List(tensors, other, alpha=2)
            return [item + item for item in result]

        tensors = [self._generate_tensor(shape, dtype, min=-8, max=8) for _ in range(2)]
        other = [self._generate_tensor(shape, dtype, min=-8, max=8) for _ in range(2)]
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensors, other)
```
