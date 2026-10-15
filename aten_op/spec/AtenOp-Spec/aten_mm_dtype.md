# aten.mm.dtype

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.mm.dtype(Tensor self, Tensor mat2, ScalarType out_dtype) -> Tensor
```

### 1.2 算子功能

矩阵乘法，计算 self @ mat2。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `mat2` | `Tensor` | 右矩阵。 |
| `out_dtype` | `int` | 输出数据类型。 |
| `返回值` | `Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | x | x | x | - |
| NPU | - | x | - | - | - | x | - | x | x | x | x | - |

#### 2.2.2 Shape 支持

`self` 和 `mat2` 固定为 2 维 Tensor。

### 2.3 特殊限制说明

输入矩阵 dtype 相同，收缩维匹配。由于社区版也不提供融合模板，所以该算子无法生成triton dsl，只会生成extern_kernels.mm_dtype下发。

### 2.4 使用方法

```python
    shapes = [(128, 128)]
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_mm_dtype(self, shape, dtype):

        def op_calc(first_element, second_element, output_dtype):
            return torch.ops.aten.mm.dtype(first_element, second_element, output_dtype)

        def op_calc_1(first_element, second_element, output_dtype):
            y = torch.ops.aten.mm.dtype(first_element, second_element, output_dtype)
            return y + y
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        right_shape = tuple(shape[:-2]) + (shape[-1], shape[-2])
        tensor2 = self._generate_tensor(right_shape, dtype, min=-2, max=2)
        output_shape = tuple(shape[:-2]) + (shape[-2], shape[-2])
        output_dtype = torch.float32
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, tensor2, output_dtype,
        )
```
