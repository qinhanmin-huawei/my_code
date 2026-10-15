# aten.empty_like

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
```
### 1.2 算子功能
`aten.empty_like` 创建一个与输入张量具有相同形状的新张量，不初始化元素值（内容为未定义）。可以通过 `dtype`、`layout`、`device`、`pin_memory` 和 `memory_format` 参数覆盖对应属性。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 参考张量，决定输出形状 |
| `dtype` | `ScalarType?` | 可选，输出张量的数据类型，默认为 `self.dtype` |
| `layout` | `Layout?` | 可选，输出张量的布局，默认为 `self.layout` |
| `device` | `Device?` | 可选，输出张量的设备，默认为 `self.device` |
| `pin_memory` | `bool?` | 可选，是否使用固定内存 |
| `memory_format` | `MemoryFormat?` | 可选，输出张量的内存格式，默认为 `Preserve` |
| 返回值 | `Tensor` | 与 `self` 形状相同的新张量，内容未初始化 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、输出形状与输入 `self` 形状相同，支持任意维度，维度范围:[1-4]

### 2.3 特殊限制说明

> 无

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64','float16', 'bfloat16', 'float32','bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_empty_like(self, shape, dtype):
        def op_calc(input):
            y = torch.ops.aten.empty_like(input)
            return y
        def op_calc_1(input):
            y = torch.ops.aten.empty_like(input)
            return y + y
        input = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input)
```
