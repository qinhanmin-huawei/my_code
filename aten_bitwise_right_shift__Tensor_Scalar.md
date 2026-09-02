# aten.bitwise_right_shift_.Tensor_Scalar

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.bitwise_right_shift_.Tensor_Scalar(self: Tensor(a!), other: Tensor) -> Tensor(a!)
```
### 1.2 算子功能
`aten.bitwise_right_shift_.Tensor_Scalar` 对输入张量 `self` 的每个元素，按**算术右移**（对于有符号整型）或**逻辑右移**（对于无符号整型）移动 `other` 指定的位数，并将结果**原地写回** `self`。

- 对于有符号整数：执行算术右移，高位用符号位填充。
- 对于无符号整数：执行逻辑右移，高位补0。
- `other` 会被转换为与 `self` 元素相同类型的整数，并作为移位量（应为非负整数）。
## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self` | `Tensor(a!)` | 输入张量，**会被原地修改**。须为整型（integral）类型 |
| `other` | `Scalar` | 移位量，必须为 Python 整数或布尔值（`True` 视为1，`False` 视为0） |
| 返回值 | `Tensor(a!)` | 返回修改后的 `self` 张量 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | ×      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | ×      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_aten_bitwise_right_shift__Tensor_Scalar(self, shape, dtype):
        def op_calc(first_element):
            torch.ops.aten.bitwise_right_shift_.Tensor_Scalar(first_element, 0b0010)
            return first_element
        def op_calc_1(first_element):
            torch.ops.aten.bitwise_right_shift_.Tensor_Scalar(first_element, 0b0010)
            return first_element + first_element
        first_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)
```
