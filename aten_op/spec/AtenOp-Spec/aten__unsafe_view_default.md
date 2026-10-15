# aten._unsafe_view.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._unsafe_view.default(Tensor self, SymInt[] size) -> Tensor
```

### 1.2 算子功能

按目标形状解释连续输入，保持元素数量不变。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `size` | `List[int]` | 目标形状。 |
| `返回值` | `Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

输入 Tensor 支持 1～4 维。

### 2.3 特殊限制说明

只覆盖连续输入及展平形状。该算子可能只修改元数据，不生成独立 kernel。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__unsafe_view_default(self, shape, dtype):

        def op_calc(first_element, size):
            return torch.ops.aten._unsafe_view.default(first_element, size)

        def op_calc_1(first_element, size):
            y = torch.ops.aten._unsafe_view.default(first_element, size)
            return y + y
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        size = (-1,)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, size,
        )
```
