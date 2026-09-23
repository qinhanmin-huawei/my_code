# aten._foreach_mul_.Scalar

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._foreach_mul_.Scalar(self: Tensor(a!)[], scalar: Scalar) -> ()
```

### 1.2 算子功能

将输入列表中的每个张量乘以同一个标量，并将结果原地写回 `self`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor[]` | 被原地修改的张量列表。 |
| `scalar` | `Scalar` | 与每个输入张量相乘的标量。 |
| `返回值` | `()` | 算子原地修改 `self`，不直接返回张量。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1. `shape` 遍历值：`(4,)`、`(4, 8)`、`(4, 8, 16)`、`(4, 8, 16, 32)`。
2. 输入列表长度固定为 `2`，每个元素使用相同的遍历 shape；bool 输入值为 `False` 或 `True` 且 `scalar` 为 `True`，其他 dtype 的输入值范围为 `[-4, 4)` 且 `scalar` 为 `2`。

### 2.3 特殊限制说明

原地结果必须可转换为 `self` 的 dtype；bool 输入使用 bool 标量并执行对应的逻辑与运算。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__foreach_mul__scalar(self, shape, dtype):
        scalar = True if dtype == 'bool' else 2

        def op_calc(tensors):
            torch.ops.aten._foreach_mul_.Scalar(tensors, scalar)
            return tensors

        def op_calc_1(tensors):
            torch.ops.aten._foreach_mul_.Scalar(tensors, scalar)
            return [item + item for item in tensors]

        tensors = [self._generate_tensor(shape, dtype, min=-4, max=4) for _ in range(2)]
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensors)
```
