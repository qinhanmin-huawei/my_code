# aten.any.dimname

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.any.dimname(self: Tensor, dim: str, keepdim: bool=False) -> Tensor
```

### 1.2 算子功能

沿指定名称的维度判断是否存在非零元素。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 带维度名称的输入张量。 |
| `dim` | `str` | 参与归约的维度名称。 |
| `keepdim` | `bool` | 是否保留归约维度。 |
| `返回值` | `Tensor` | bool 类型的归约结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | x | x | x | x |
| NPU | - | x | - | - | - | x | - | x | x | x | x | x |


#### 2.2.2 Shape 支持

输入 Tensor 支持 1～4 维。

### 2.3 特殊限制说明

dimname类型用例无法构建输入。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_any_dimname(self, shape, dtype):
        def op_calc(first_element):
            return torch.ops.aten.any.dimname(first_element, 'reduce', True)

        def op_calc_1(first_element):
            result = torch.ops.aten.any.dimname(first_element, 'reduce', True)
            return torch.logical_or(result, result)

        tensor1 = self._generate_tensor(shape, dtype, min=-1, max=1)
        names = (None,) * (len(shape) - 1) + ('reduce',)
        tensor1 = tensor1.refine_names(*names)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1)
```
