# aten.adaptive_max_pool2d.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.adaptive_max_pool2d.default(Tensor self, int[2] output_size) -> (Tensor, Tensor)
```

### 1.2 算子功能

自适应最大池化，返回 2 维空间池化结果及索引。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `output_size` | `List[int]` | 输出空间大小。 |
| `返回值` | `Tensor, Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |
| NPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |

#### 2.2.2 Shape 支持

`self` 支持 3～4 维 Tensor。

### 2.3 特殊限制说明

输入使用 (C, H, W) 或 (N, C, H, W)。使用 aclnnAdaptiveMaxPool2d，索引为 int64。当输入必须满足：ceil(H/output_size[0]) * ceil (W/output_size[1]) <= 25时才可以正确融合。

### 2.4 使用方法

```python
    shapes = [(4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_adaptive_max_pool2d_default(self, shape, dtype):

        def op_calc(first_element, output_size):
            y = torch.ops.aten.adaptive_max_pool2d.default(first_element, output_size)
            return (y[0], y[1])

        def op_calc_1(first_element, output_size):
            y = torch.ops.aten.adaptive_max_pool2d.default(first_element, output_size)
            y = (y[0], y[1])
            return (y[0] + y[0], *y[1:])
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        output_size = (3, 3)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, output_size,
        )
```
