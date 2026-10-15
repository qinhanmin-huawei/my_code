# aten.bitwise_or.Scalar_Tensor

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.bitwise_or.Scalar_Tensor(self: Scalar, other: Tensor) -> Tensor
```
### 1.2 算子功能
`aten.bitwise_or.Scalar_Tensor` 计算**标量（Scalar）** 与输入张量之间的按位或（Bitwise OR）运算。即对标量值的二进制位与张量中每一个元素的二进制位进行逐位或运算。

- **整型按位或**：将标量整数的二进制位与张量中每个整数的二进制位逐位进行或运算。规则为：只要有一个对应位为 `1`，结果位即为 `1`；否则为 `0`。
- **布尔型逻辑或**：如果输入张量是布尔型，算子执行逻辑或（Logical OR）运算（`True` 或 `False`）。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self` | `Scalar`（`int` / `bool`） | 用于按位或运算的标量值（Python 整数或布尔值） |
| `other` | `Tensor` | 输入的源张量，须为整型或布尔类型 |
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

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_bitwise_or_Scalar_Tensor(self, shape, dtype):
        def op_calc(tensor):
            y = torch.ops.aten.bitwise_or.Scalar_Tensor(5, tensor)
            return y
        def op_calc_1(tensor):
            y = torch.ops.aten.bitwise_or.Scalar_Tensor(5, tensor)
            return y + y
        tensor = self._generate_tensor(shape, dtype)
        
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor)
```
