# aten.diagonal_scatter.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.diagonal_scatter.default(self: Tensor, src: Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> Tensor
```
### 1.2 算子功能
`aten.diagonal_scatter.default` 是 `aten.diagonal_scatter` 的默认重载（default overload），其功能与 `aten.diagonal_scatter` 完全一致：将 `src` 张量的元素写入 `self` 中由 `offset`、`dim1`、`dim2` 确定的对角线位置，并返回一个全新张量，不原地修改 `self`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 目标源张量，需要至少 2 维 |
| `src` | `Tensor` | 待写入对角线位置的源数据张量 |
| `offset` | `int` | 对角线偏移量，默认为 0（主对角线） |
| `dim1` | `int` | 参与写入的第一个维度，默认为 0 |
| `dim2` | `int` | 参与写入的第二个维度，默认为 1 |
| 返回值 | `Tensor` | 写入后的新张量，形状与 `self` 一致，不修改 `self` |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、shape维度范围：[2-4]（需至少 2 维，且 `dim1` 与 `dim2` 不同）


### 2.3 特殊限制说明

> `dim1`、`dim2` 需为有效且不同的维度。

### 2.4 使用方法

```python
    def generate_src(self, input):
        return torch.zeros_like(input.diagonal())

    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_diagonal_scatter_default(self, shape, dtype):
        def op_calc(input, src):
            y = torch.ops.aten.diagonal_scatter.default(input, src)
            return y
        def op_calc_1(input, src):
            y = torch.ops.aten.diagonal_scatter.default(input, src)
            return y + y
        input = self._generate_tensor(shape, dtype)
        src = self.generate_src(input)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input, src)
```
