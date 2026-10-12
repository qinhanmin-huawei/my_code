# aten.fractional_max_pool3d

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)
```

### 1.2 算子功能

分数最大池化，返回 3 维空间池化结果及索引。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `kernel_size` | `List[int]` | 池化窗口大小。 |
| `output_size` | `List[int]` | 输出空间大小。 |
| `random_samples` | `Tensor` | 池化区域随机样本，取值范围 [0, 1)。 |
| `返回值` | `Tensor, Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |
| NPU | - | x | - | - | - | x | - | x | x | x | x | - |

#### 2.2.2 Shape 支持

`self` 支持 4～5 维 Tensor；`random_samples` 固定为 3 维 Tensor。

### 2.3 特殊限制说明

输入使用四维 (C, D, H, W)，random_samples 的 shape 为 (1, C, 3)，由框架生成，范围 [0, 1)；eager 与编译调用使用同一份样本。索引输出为 int64。 当前 Ascend 未注册 fractional_max_pool3d 原生实现，Inductor 将其列为 fallback；NPU 列不将 CPU fallback 计作原生支持。

### 2.4 使用方法

```python
    shapes = [(4, 8, 16, 32)]
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_fractional_max_pool3d(self, shape, dtype):
        def op_calc(first_element, kernel_size, output_size, random_samples):
            y = torch.ops.aten.fractional_max_pool3d(first_element, kernel_size, output_size, random_samples)
            return (y[0], y[1])

        def op_calc_1(first_element, kernel_size, output_size, random_samples):
            y = torch.ops.aten.fractional_max_pool3d(first_element, kernel_size, output_size, random_samples)
            y = (y[0], y[1])
            return (y[0] + y[0], *y[1:])
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        kernel_size = (2, 2, 2)
        output_size = (3, 3, 3)
        random_samples = self._generate_tensor((1, shape[0], 3), dtype, min=0, max=1)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, kernel_size, output_size, random_samples,
        )
```
