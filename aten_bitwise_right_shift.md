# aten.bitwise_right_shift

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.bitwise_right_shift(self: Tensor, other: Tensor) -> Tensor
```
### 1.2 算子功能
`aten.bitwise_right_shift` 计算输入张量 `self` 中每个元素**按位右移** `other` 位，并返回一个新的张量。

- **整型按位右移**：将 `self` 中每个整数的二进制位向右移动 `other` 位（或 `other` 张量对应元素的位数）。
  - 对于**有符号整数类型**，执行**算术右移**（保留符号位，高位用符号位填充）。
  - 对于**无符号整数类型**，执行**逻辑右移**（高位补0）。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 输入张量，须为整型（integral）类型 |
| `other` | `Tensor` | 须与 `self` 类型兼容且可广播|
| 返回值 | `Tensor` | 一个新的张量，形状为广播后的形状，包含右移后的结果 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | ×      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_bitwise_right_shift(self, shape, dtype):
        def op_calc(first_element, second_element):
            y = torch.ops.aten.bitwise_right_shift(first_element, second_element)
            return y
        def op_calc_1(first_element, second_element):
            y = torch.ops.aten.bitwise_right_shift(first_element, second_element)
            return y + y
        first_element = self._generate_tensor(shape, dtype)
        second_element = torch.full(shape, 2, dtype=torch.int32, device="npu")
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, second_element)
```
