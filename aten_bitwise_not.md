# aten.bitwise_not

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.bitwise_not(self: Tensor) -> Tensor
```
### 1.2 算子功能
`aten.bitwise_not` 
*   **整型按位翻转**：对输入张量中每个整数的二进制位进行逐位翻转（`0` 变 `1`，`1` 变 `0`）。根据计算机补码原理，对于有符号整数，`~x` 的数学结果等于 `-(x + 1)`。
*   **布尔型逻辑反转**：如果输入是布尔矩阵，算子会将所有 `True` 转换为 `False`，`False` 转换为 `True`。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self/input`        | `Tensor`          | 输入的源张量 |
| `out`(返回值)      | `Tensor`             | 一个新的张量，形状与输入一致，包含计算后的结果 |


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
    dtypes = ['int8', 'int32', 'int64', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_bitwise_not(self, shape, dtype):
        def op_calc(first_element):
            y = torch.ops.aten.bitwise_not(first_element)
            return y

        def op_calc_1(first_element):
            y = torch.ops.aten.bitwise_not(first_element)
            return y + y
        first_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)
```
