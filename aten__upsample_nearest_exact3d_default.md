# aten._upsample_nearest_exact3d.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._upsample_nearest_exact3d.default(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
```

### 1.2 算子功能

对 3 维空间执行精确最近邻上采样。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `output_size` | `List[int]` | 输出空间大小。 |
| `scales_d` | `Optional[float]` | 算子配置参数，取值见使用方法。 |
| `scales_h` | `Optional[float]` | 算子配置参数，取值见使用方法。 |
| `scales_w` | `Optional[float]` | 算子配置参数，取值见使用方法。 |
| `返回值` | `Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |
| NPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |

#### 2.2.2 Shape 支持

1. 本测试覆盖的 shape 维度固定为 `5`。
2. 测试 shape：`(1, 4, 8, 16, 32)`。

### 2.3 特殊限制说明

输入为 N、C 和空间维；通过 output_size 指定输出，scales 保持默认 None。

### 2.4 使用方法

```python
    shapes = [(1, 4, 8, 16, 32)]
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__upsample_nearest_exact3d_default(self, shape, dtype):

        def op_calc(first_element, output_size):
            return torch.ops.aten._upsample_nearest_exact3d.default(first_element, output_size)

        def op_calc_1(first_element, output_size):
            y = torch.ops.aten._upsample_nearest_exact3d.default(first_element, output_size)
            return y + y
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        output_size = tuple(s * 2 for s in shape[-3:])
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, output_size,
        )
```
