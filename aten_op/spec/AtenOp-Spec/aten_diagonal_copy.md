# aten.diagonal_copy

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.diagonal_copy(self: Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> Tensor
```
### 1.2 算子功能
`aten.diagonal_copy` 的核心功能是提取输入张量 `self` 中由 `dim1`、`dim2` 两个维度与 `offset` 共同确定的对角线元素，并拷贝到一个全新的张量中返回。该算子不共享底层存储，与视图语义的 `diagonal` 不同，返回结果与原张量相互独立。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 源张量，需要至少 2 维 |
| `offset` | `int` | 对角线偏移量，默认为 0（主对角线）；正值取主对角线右上方，负值取左下方 |
| `dim1` | `int` | 参与取对角的第一个维度，默认为 0 |
| `dim2` | `int` | 参与取对角的第二个维度，默认为 1 |
| 返回值 | `Tensor` | 提取对角线后生成的新张量，与原张量不共享存储 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、shape维度范围：[2-4]（需至少 2 维，且 `dim1` 与 `dim2` 不同）


### 2.3 特殊限制说明

> `dim1` 和 `dim2` 必须指向不同的有效维度，且都在 `[-self.dim(), self.dim())` 范围内。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_diagonal_copy(self, shape, dtype):
        def op_calc(input):
            y = torch.ops.aten.diagonal_copy(input, offset = 0, dim1 = 0, dim2 = 1)
            return y
        def op_calc_1(input):
            y = torch.ops.aten.diagonal_copy(input, offset = 0, dim1 = 0, dim2 = 1)
            return y + y
        input = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input)
```
