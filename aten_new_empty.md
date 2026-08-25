# aten.new_empty

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.new_empty(
    Tensor self, 
    SymInt[] size, 
    *, 
    ScalarType? dtype=None, 
    Layout? layout=None, 
    Device? device=None, 
    bool? pin_memory=None
) -> Tensor
```
### 1.2 算子功能
`aten.new_empty` 创建一个指定形状的新张量，不初始化元素值。新张量的 `dtype`、`layout`、`device` 默认与输入张量 `self` 相同，也可通过可选参数覆盖。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 参考张量，决定默认的 dtype、layout、device |
| `size` | `SymInt[]` | 输出张量的形状 |
| `dtype` | `ScalarType?` | 可选，输出张量的数据类型 |
| `layout` | `Layout?` | 可选，输出张量的布局 |
| `device` | `Device?` | 可选，输出张量的设备 |
| `pin_memory` | `bool?` | 可选，是否使用固定内存 |
| 返回值 | `Tensor` | 指定形状的新张量，内容未初始化 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ |  |
| NPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ |  |

#### 2.2.2 Shape 支持

1、输出形状由 `size` 参数指定，支持任意维度

### 2.3 特殊限制说明

> 无

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64','float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_new_empty(self, shape, dtype):
        def op_calc(input, size):
            y = torch.ops.aten.new_empty(input, size)
            return y
        def op_calc_1(input, size):
            y = torch.ops.aten.new_empty(input, size)
            return y + y
        input = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input, shape)
```
