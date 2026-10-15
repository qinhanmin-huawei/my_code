# aten.baddbmm.dtype_out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.baddbmm.dtype_out(Tensor self, Tensor batch1, Tensor batch2, ScalarType out_dtype, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能

批量矩阵乘法，计算 beta × self + alpha × mat1 @ mat2。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `batch1` | `Tensor` | 左侧批量矩阵。 |
| `batch2` | `Tensor` | 右侧批量矩阵。 |
| `out_dtype` | `int` | 输出数据类型。 |
| `beta` | `number` | 加数系数。 |
| `alpha` | `number` | 矩阵乘结果系数。 |
| `out` | `Tensor` | 输出缓冲区。 |
| `返回值` | `Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | x | x | x | - |
| NPU | - | x | - | - | - | x | - | x | x | x | x | - |

#### 2.2.2 Shape 支持

`batch1` 和 `batch2` 固定为 3 维 Tensor；`self` 支持 0～3 维 Tensor。

### 2.3 特殊限制说明

在fake tensor层级无法匹配output dtype这个参数，导致直接break graph，这是社区v2.10.0pytorch存在的预编译bug，issue链接:https://github.com/pytorch/pytorch/issues/163880。

### 2.4 使用方法

```python
    shapes = [(4, 8, 16)]
    dtypes = ['float16', 'bfloat16', 'float32']
    coefficients = [(1, 1), (0, 2)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    @parametrize('coefficients', coefficients)
    def test_aten_baddbmm_dtype_out(self, shape, dtype, coefficients):

        def op_calc(bias, first_element, second_element, output_dtype, beta, alpha, out):
            return torch.ops.aten.baddbmm.dtype_out(bias, first_element, second_element, output_dtype, beta=beta, alpha=alpha, out=out)

        def op_calc_1(bias, first_element, second_element, output_dtype, beta, alpha, out):
            y = torch.ops.aten.baddbmm.dtype_out(bias, first_element, second_element, output_dtype, beta=beta, alpha=alpha, out=out)
            return y + y
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        right_shape = tuple(shape[:-2]) + (shape[-1], shape[-2])
        tensor2 = self._generate_tensor(right_shape, dtype, min=-2, max=2)
        output_shape = tuple(shape[:-2]) + (shape[-2], shape[-2])
        output_dtype = torch.float32
        bias = self._generate_tensor(output_shape, dtype, min=-2, max=2)
        beta, alpha = coefficients
        out = self._generate_tensor(output_shape, 'float32', min=-2, max=2)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, bias, tensor1, tensor2, output_dtype, beta, alpha, out,
        )
```
