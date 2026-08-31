# aten.as_strided_scatter.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.as_strided_scatter.default(self: Tensor, src: Tensor, size: int[], stride: int[], storage_offset: int = None) -> Tensor
```
### 1.2 算子功能
`aten.as_strided_scatter.default` 是 `aten.as_strided_scatter` 的默认重载（default overload），其功能与 `aten.as_strided_scatter` 一致：将 `src` 张量按照给定的 `size`、`stride`、`storage_offset` 以视图（as_strided）的布局方式写入 `self` 对应位置，并返回一个全新张量，不原地修改 `self`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 目标源张量，作为写入的底层存储 |
| `src` | `Tensor` | 待写入的源数据张量 |
| `size` | `int[]` | 视图形状，长度与 `stride` 一致 |
| `stride` | `int[]` | 视图在每个维度上的步长 |
| `storage_offset` | `int` | 写入区域在底层存储中的偏移量，默认为 None（0） |
| 返回值 | `Tensor` | 写入后的新张量，形状与 `self` 一致，不修改 `self` |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、`size` 与 `stride` 长度必须相等，维度范围：[1-4]

### 2.3 特殊限制说明

> `size` 与 `stride` 的长度必须一致

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_as_strided_scatter_default(self, shape, dtype):
        def op_calc(tensor, src_tensor, new_size, new_stride):
            y = torch.ops.aten.as_strided_scatter.default(tensor, src_tensor, new_size, new_stride, storage_offset = 0)
            return y
        def op_calc_1(tensor, src_tensor, new_size, new_stride):
            y = torch.ops.aten.as_strided_scatter.default(tensor, src_tensor, new_size, new_stride, storage_offset = 0)
            return y + y
        tensor = self._generate_tensor(shape, dtype)
        new_size = (tensor.numel()//64, 64)
        new_stride = (64, 1)
        src_tensor = self._generate_tensor(new_size, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, src_tensor, new_size, new_stride)
```
