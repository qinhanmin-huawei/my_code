# aten.bitwise_right_shift.Tensor_Scalar

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.bitwise_right_shift.Tensor_Scalar(self: Tensor, other: Scalar) -> Tensor
```
### 1.2 算子功能
`aten.bitwise_right_shift.Tensor_Scalar` 对输入张量 `self` 中每个元素进行按位右移，移动位数由标量 `other` 指定，并返回新的张量。

- **整型按位右移**：
  - 对于**有符号整数类型**（如 `int8`、`int32`），执行**算术右移**（保留符号位，高位用符号位填充）。
  - 对于**无符号整数类型**（如 `uint8`、`uint32`），执行**逻辑右移**（高位补0）。
- **布尔型**：如果输入张量是布尔类型，`True` 视为1，`False` 视为0，执行右移后结果仍为整型（通常提升为 `uint8` 或 `int64`）。
- 标量值会被转换为与张量元素相同的类型，然后作为移位量应用。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 被右移的张量，须为整型或布尔类型 |
| `other` | `Scalar`（`int` / `bool`） | 右移位数，须为非负整数；布尔值 `True` 视为1，`False` 视为0 |
| 返回值 | `Tensor` | 新的张量，形状与 `self` 相同，包含右移后的结果 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | √      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', "bool"]
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_bitwise_right_shift_Tensor_Scalar(self, shape, dtype):
        def op_calc(first_element):
            y = torch.ops.aten.bitwise_right_shift.Tensor_Scalar(first_element, 0b1100)
            return y
        def op_calc_1(first_element):
            y = torch.ops.aten.bitwise_right_shift.Tensor_Scalar(first_element, 0b1100)
            return y + y
        first_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)

```
