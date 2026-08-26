# aten.as_stride_copy.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.as_stride_copy.out(self: Tensor, size: int[], stride: int[], storage_offset: int, *, out: Tensor) -> Tensor
```
### 1.2 算子功能
`aten.as_stride_copy.out` 根据指定的 `size`、`stride` 和 `storage_offset` 对输入张量进行重解释，并将结果写入预先分配的 `out` 张量中。结果张量不与原张量共享存储。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 源张量 |
| `size` | `int[]` | 目标形状 |
| `stride` | `int[]` | 目标步长 |
| `storage_offset` | `int` | 存储偏移量 |
| `out` | `Tensor` | 输出张量，其形状必须与 `size` 参数一致，数据类型和设备必须与 `self` 匹配 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`size` 和 `stride` 的长度必须相等

### 2.3 特殊限制说明

> 指定的 `size` 和 `stride` 组合必须保证在存储范围内有效，即 `storage_offset + sum((size[i] - 1) * stride[i])` 不超过 `self` 的存储大小。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_as_stride_copy_out(self, shape, dtype):
        def op_calc(input, out_t):
            size = list(input.shape)
            stride = list(input.stride())
            torch.ops.aten.as_stride_copy.out(input, size, stride, out=out_t)
            return out_t
        def op_calc_1(input, out_t):
            size = list(input.shape)
            stride = list(input.stride())
            torch.ops.aten.as_stride_copy.out(input, size, stride, out=out_t)
            return out_t + out_t
        out_t = torch.empty(shape, device='npu', dtype=eval("torch." + dtype))
        input = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input, out_t)
```