# aten.max_pool3d_with_indices.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.max_pool3d_with_indices.default(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
```

### 1.2 算子功能

滑动窗口最大池化，返回 3 维空间池化结果及索引。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `kernel_size` | `List[int]` | 池化窗口大小。 |
| `stride` | `List[int]` | 算子配置参数，取值见使用方法。 |
| `padding` | `List[int]` | 算子配置参数，取值见使用方法。 |
| `dilation` | `List[int]` | 算子配置参数，取值见使用方法。 |
| `ceil_mode` | `bool` | 算子配置参数，取值见使用方法。 |
| `返回值` | `Tensor, Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |
| NPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |

#### 2.2.2 Shape 支持

`self` 支持 4～5 维 Tensor。

### 2.3 特殊限制说明

输入使用四维 (C, D, H, W) 非 batch 形式。NPU max_pool 索引使用 int32；2d 开启兼容实现以返回普通索引，结束后恢复设置。

### 2.4 使用方法

```python
    shapes = [(4, 8, 16, 32)]
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_max_pool3d_with_indices_default(self, shape, dtype):

        def op_calc(first_element, kernel_size, stride, padding, dilation, ceil_mode):
            y = torch.ops.aten.max_pool3d_with_indices.default(first_element, kernel_size, stride, padding, dilation, ceil_mode)
            return (y[0], y[1].to(torch.int64))

        def op_calc_1(first_element, kernel_size, stride, padding, dilation, ceil_mode):
            y = torch.ops.aten.max_pool3d_with_indices.default(first_element, kernel_size, stride, padding, dilation, ceil_mode)
            y = (y[0], y[1])
            return (y[0] + y[0], *y[1:].to(torch.int64))
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        kernel_size = (2, 2, 2)
        stride = (2, 2, 2)
        padding = (0, 0, 0)
        dilation = (1, 1, 1)
        ceil_mode = False
        output_size = tuple((s + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1 for i, s in enumerate(shape[-3:]))
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, kernel_size, stride, padding, dilation, ceil_mode,
        )
```
