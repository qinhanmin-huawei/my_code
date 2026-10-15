# aten._foreach_mul.Scalar

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._foreach_mul.Scalar(self: Tensor[], scalar: Scalar) -> Tensor[]
```

### 1.2 算子功能

将输入列表中的每个张量乘以同一个标量，计算 `self[i] * scalar`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor[]` | 输入张量列表。 |
| `scalar` | `Scalar` | 与每个输入张量相乘的标量。 |
| `返回值` | `Tensor[]` | 与输入列表等长的结果列表。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

注：npu eager模式不支持int64，int8，bool类型。
#### 2.2.2 Shape 支持

输入列表中的 Tensor 支持 1～4 维。

### 2.3 特殊限制说明

标量必须可转换为输入张量支持的计算类型；bool 输入使用 bool 标量并执行对应的逻辑与运算。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__foreach_mul_scalar(self, shape, dtype):
        scalar = True if dtype == 'bool' else 2

        def op_calc(tensors):
            return torch.ops.aten._foreach_mul.Scalar(tensors, scalar)

        def op_calc_1(tensors):
            y = torch.ops.aten._foreach_mul.Scalar(tensors, scalar)
            return [item + item for item in y]

        tensors = [self._generate_tensor(shape, dtype, min=-4, max=4) for _ in range(2)]
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensors)
```
