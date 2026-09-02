# aten.empty_strided.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.empty_strided.default(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
```
### 1.2 算子功能
`aten.empty_strided.default` 是 `aten.empty_strided` 的默认 overload，功能相同：创建一个指定形状和步长的新张量，不初始化元素值。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `size` | `SymInt[]` | 输出张量的形状 |
| `stride` | `SymInt[]` | 输出张量在每个维度上的步长 |
| `dtype` | `ScalarType?` | 可选，输出张量的数据类型 |
| `layout` | `Layout?` | 可选，输出张量的布局 |
| `device` | `Device?` | 可选，输出张量的设备 |
| `pin_memory` | `bool?` | 可选，是否使用固定内存 |
| 返回值 | `Tensor` | 指定形状和步长的新张量，内容未初始化 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、`size` 和 `stride` 长度必须相等，维度范围：[1-4]

### 2.3 特殊限制说明

> stride 的长度必须与 size 相同。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64','float16', 'bfloat16', 'float32','bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_empty_strided_default(self, shape, dtype):
        # 根据shape计算contiguous布局的stride
        stride = []
        s = 1
        for dim in reversed(shape):
            stride.append(s)
            s *= dim
        stride = tuple(reversed(stride))
        dtype_t = eval("torch." + dtype)
        device_t = 'npu'
        def op_calc(size, stride, dtype_t, device_t):
            y = torch.ops.aten.empty_strided.default(size, stride, dtype=dtype_t, device=device_t)
            return y
        def op_calc_1(size, stride, dtype_t, device_t):
            y = torch.ops.aten.empty_strided.default(size, stride, dtype=dtype_t, device=device_t)
            return y + y
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, shape, stride, dtype_t, device_t)
```
