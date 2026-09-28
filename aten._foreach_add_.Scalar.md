# aten._foreach_add_.Scalar

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._foreach_add_.Scalar(self: Tensor(a!)[], scalar: Scalar) -> ()
```

### 1.2 算子功能

将同一个标量原地加到输入列表中的每个张量。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor[]` | 被原地修改的张量列表。 |
| `scalar` | `Scalar` | 加到每个输入张量上的标量。 |
| `返回值` | `()` | 算子原地修改 `self`。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

注：bool 输入执行逻辑或语义，npu eager模式不支持int64，int8，bool类型。
#### 2.2.2 Shape 支持

输入列表中的 Tensor 支持 1～4 维。

### 2.3 特殊限制说明

标量和计算结果必须能够转换并原地写回输入 dtype。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__foreach_add__scalar(self, shape, dtype):
        scalar = True if dtype == 'bool' else 2

        def op_calc(tensors):
            torch.ops.aten._foreach_add_.Scalar(tensors, scalar)
            return torch.stack(tensors) if dtype == 'bool' else tensors

        def op_calc_1(tensors):
            torch.ops.aten._foreach_add_.Scalar(tensors, scalar)
            if dtype == 'bool':
                result = torch.stack(tensors)
                return torch.logical_or(result, result)
            return [item + item for item in tensors]

        tensors = [self._generate_tensor(shape, dtype, min=-4, max=4) for _ in range(2)]
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensors)
```
