# aten.amin.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.amin.default(self: Tensor, dim: int[1]=[], keepdim: bool=False) -> Tensor
```

### 1.2 算子功能

沿指定维度计算输入张量的最小值。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量。 |
| `dim` | `int[1]` | 归约维度列表。 |
| `keepdim` | `bool` | 是否保留归约维度。 |
| `返回值` | `Tensor` | 归约结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |


#### 2.2.2 Shape 支持

输入 Tensor 支持 1～4 维。

### 2.3 特殊限制说明

NPU 不覆盖 bool 输入；不覆盖复数。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_amin_default(self, shape, dtype):
        def op_calc(first_element):
            return torch.ops.aten.amin.default(first_element, (-1,), True)

        def op_calc_1(first_element):
            result = torch.ops.aten.amin.default(first_element, (-1,), True)
            return result + result

        tensor1 = self._generate_tensor(shape, dtype, min=-8, max=8)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1)
```
