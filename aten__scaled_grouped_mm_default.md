# aten._scaled_grouped_mm.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._scaled_grouped_mm.default(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? offs=None, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False) -> Tensor
```

### 1.2 算子功能

执行带缩放因子的分组矩阵乘法。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `mat2` | `Tensor` | 右矩阵。 |
| `scale_a` | `Tensor` | 左输入缩放因子。 |
| `scale_b` | `Tensor` | 右输入缩放因子。 |
| `offs` | `Optional[Tensor]` | 算子配置参数，取值见使用方法。 |
| `bias` | `Optional[Tensor]` | 算子配置参数，取值见使用方法。 |
| `scale_result` | `Optional[Tensor]` | 算子配置参数，取值见使用方法。 |
| `out_dtype` | `Optional[int]` | 输出数据类型。 |
| `use_fast_accum` | `bool` | 算子配置参数，取值见使用方法。 |
| `返回值` | `Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | x | x | x | - |
| NPU | - | x | - | - | - | x | - | x | x | x | x | - |

#### 2.2.2 Shape 支持

1. 本测试覆盖的 shape 维度固定为 `3`。
2. 测试 shape：`(1, 128, 128)`。

### 2.3 特殊限制说明

eager模式在npu上和cpu上均无法运行，该算子只支持在npu上下发。且输入要求为fp8，npu triton还不支持fp8。
### 2.4 使用方法

```python
    shapes = [(1, 128, 128)]
    dtypes = []

    @unittest.skip('该算子的 GPU 正向输入类型超出本次七种 dtype 范围，NPU 输入组合尚未确认')
    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__scaled_grouped_mm_default(self, shape, dtype):

        def op_calc(first_element, second_element, scale_a, scale_b, options):
            return torch.ops.aten._scaled_grouped_mm.default(first_element, second_element, scale_a, scale_b, **options)

        def op_calc_1(first_element, second_element, scale_a, scale_b, options):
            y = torch.ops.aten._scaled_grouped_mm.default(first_element, second_element, scale_a, scale_b, **options)
            return y + y
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        tensor2 = self._generate_tensor(shape, dtype, min=-2, max=2)
        tensor2 = tensor2.transpose(-1, -2)
        scale_a = self._generate_tensor(shape[:-1], 'float32', min=1, max=2)
        scale_b = self._generate_tensor(shape[:-1], 'float32', min=1, max=2)
        options = {
            'offs': None, 'bias': None, 'scale_result': None,
            'out_dtype': torch.bfloat16, 'use_fast_accum': False,
        }
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, tensor2, scale_a, scale_b, options,
        )
```
