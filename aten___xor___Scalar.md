# aten.__xor__.Scalar

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.__xor__.Scalar(Tensor self, Scalar other) -> Tensor
```

### 1.2 算子功能

逐元素按位异或。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `other` | `number` | 第二个操作数。 |
| `返回值` | `Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | x | x | x | √ |
| NPU | - | √ | - | - | - | √ | - | √ | x | x | x | √ |

#### 2.2.2 Shape 支持

1. 本测试覆盖的 shape 维度范围为 `[1, 4]`。
2. 测试 shape：`(4,)`、`(4, 8)`、`(4, 8, 16)`、`(4, 8, 16, 32)`。

### 2.3 特殊限制说明

浮点输入不支持；bool 输入执行对应逻辑运算。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'bool']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten___xor___Scalar(self, shape, dtype):

        def op_calc(first_element, other):
            return torch.ops.aten.__xor__.Scalar(first_element, other)

        def op_calc_1(first_element, other):
            y = torch.ops.aten.__xor__.Scalar(first_element, other)
            return y + y
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        scalar = self._generate_random_scalar(dtype)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, scalar,
        )
```
