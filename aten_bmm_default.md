# aten.bmm.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.bmm.default(Tensor self, Tensor mat2) -> Tensor
```

### 1.2 算子功能

批量矩阵乘法，计算 self @ mat2。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `mat2` | `Tensor` | 右矩阵。 |
| `返回值` | `Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |
| NPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |

#### 2.2.2 Shape 支持

1. 输入参数 `self` 和 `mat2` 必须为三维 Tensor。
2. 测试 shape：`(4, 8, 16)`。

### 2.3 特殊限制说明

输入矩阵 dtype 相同，收缩维匹配。输出保持输入 dtype。由于走模板库的降级，需要编译时开启"max_autotune": True,"max_autotune_gemm_backends": "TRITON,ATEN","epilogue_fusion": True才能走融合。

### 2.4 使用方法

```python
    shapes = [(4, 8, 16)]
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_bmm_default(self, shape, dtype):

        def op_calc(first_element, second_element):
            return torch.ops.aten.bmm.default(first_element, second_element)

        def op_calc_1(first_element, second_element):
            y = torch.ops.aten.bmm.default(first_element, second_element)
            return y + y
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        right_shape = tuple(shape[:-2]) + (shape[-1], shape[-2])
        tensor2 = self._generate_tensor(right_shape, dtype, min=-2, max=2)
        output_shape = tuple(shape[:-2]) + (shape[-2], shape[-2])
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, tensor2,
        )
```
