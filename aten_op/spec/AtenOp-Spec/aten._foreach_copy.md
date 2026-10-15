# aten._foreach_copy

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._foreach_copy(self: Tensor[], src: Tensor[], non_blocking: bool=False) -> Tensor[]
```

### 1.2 算子功能

将 `src` 列表中的张量逐项复制到以 `self` 为模板的结果列表。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor[]` | 提供结果属性的目标张量列表。 |
| `src` | `Tensor[]` | 被复制的源张量列表。 |
| `non_blocking` | `bool` | 是否请求异步复制。 |
| `返回值` | `Tensor[]` | 复制后的结果列表。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

输入列表中的 Tensor 支持 1～4 维。

### 2.3 特殊限制说明

两个列表必须等长。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__foreach_copy(self, shape, dtype):
        def op_calc(tensors, src):
            result = torch.ops.aten._foreach_copy(tensors, src, False)
            return torch.stack(result)

        def op_calc_1(tensors, src):
            result = torch.ops.aten._foreach_copy(tensors, src, False)
            result = torch.stack(result)
            return torch.logical_or(result, result) if dtype == 'bool' else result + result

        tensors = [self._generate_tensor(shape, dtype, min=-8, max=8) for _ in range(2)]
        src = [self._generate_tensor(shape, dtype, min=-8, max=8) for _ in range(2)]
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensors, src)
```
