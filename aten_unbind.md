# aten.unbind

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.unbind(self: Tensor, dim: int = 0) -> Tensor[]
```
### 1.2 算子功能
`aten.unbind` 将输入源张量`self`沿着指定的维度轴`dim`完全拆散，移除该维度，并将其包含的每一个切片作为一个独立的子张量组成列表返回。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 源张量 |
| `dim` | `int` | 要移除的维度，默认为 0 |
| 返回值 | `Tensor[]` | 沿 `dim` 维度切分后的子张量元组，数量为 `self.shape[dim]` |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`dim` 必须在 `[-self.ndim, self.ndim - 1]` 范围内

### 2.3 特殊限制说明

> 无

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_unbind(self, shape, dtype):
        def op_calc(input):
            y = torch.ops.aten.unbind(input, dim=0)
            return y[0]
        def op_calc_1(input):
            y = torch.ops.aten.unbind(input, dim=0)
            return y[0] + 1
        input = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input)
```