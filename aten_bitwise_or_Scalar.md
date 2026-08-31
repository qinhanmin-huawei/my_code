# aten.bitwise_or.Scalar

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.bitwise_or.Scalar(self: Tensor, other: Scalar) -> Tensor
```
### 1.2 算子功能
`aten.bitwise_or.Scalar` 计算输入张量与**标量（Scalar）** 之间的按位或（Bitwise OR）运算。即对输入张量中的每一个元素，与给定的标量值进行逐位或运算。

*   **整型按位或**：将张量中每个整数的二进制位与标量整数的二进制位逐位进行或运算。规则为：只要有一个对应位为 `1`，结果位即为 `1`；否则为 `0`。
*   **布尔型逻辑或**：如果输入张量是布尔型，算子执行逻辑或（Logical OR）运算（`True` 或 `False`）。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 第一个输入张量 |
| `other` | `Scalar`（`int` / `bool`） | 用于按位或运算的标量值（Python 整数或布尔值） |
| 返回值 | `Tensor` | 一个新的张量，形状与输入一致，包含计算后的结果 |


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

    def test_aten_bitwise_or_Scalar(self, shape, dtype):
        def op_calc(tensor, Scalar):
            y = torch.ops.aten.bitwise_or.Scalar(tensor, Scalar)
            return y
        def op_calc_1(tensor, Scalar):
            y = torch.ops.aten.bitwise_or.Scalar(tensor, Scalar)
            return y + y
        tensor = self._generate_tensor(shape, dtype)
        Scalar = self._generate_random_scalar(dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, Scalar)
```
