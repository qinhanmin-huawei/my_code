# aten.bitwise_or_.Scalar

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.bitwise_or_.Scalar(self: Tensor(a!), other: Scalar) -> Tensor(a!)
```
### 1.2 算子功能
`aten.bitwise_or_.Scalar` 是 `aten.bitwise_or` 的**原地（in-place）版本**，专门针对 `other` 参数为标量（Scalar）的情况。它计算输入张量 `self` 与标量值之间的**按位或（Bitwise OR）** 运算，并将结果**直接写回** `self` 中。

- **整型按位或**：将张量中每个元素的二进制位与标量整数的二进制位逐位进行或运算。规则：两个对应位中只要有一个为 `1`，结果位即为 `1`；否则为 `0`。
- **布尔型逻辑或**：如果输入张量是布尔型，算子执行逻辑或（Logical OR）运算。
- **原地操作特性**：函数名以下划线 `_` 结尾，且后缀 `.Scalar` 表明 `other` 为标量类型，该版本不会分配新的张量，直接修改原始张量数据。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor(a!)` | 输入张量，**会被原地修改**。须为整型或布尔类型 |
| `other` | `Scalar`（`int` / `bool`） | 用于按位或运算的标量值（Python 整数或布尔值） |
| 返回值 | `Tensor(a!)` | 返回修改后的 `self` 张量本身 |


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

    def test_aten_bitwise_or__scalar(self, shape, dtype):
        def op_calc(first_element, scalar):
            y = torch.ops.aten.bitwise_or_.Scalar(first_element, scalar)
            return y
        def op_calc_1(first_element, scalar):
            y = torch.ops.aten.bitwise_or_.Scalar(first_element, scalar)
            return y + 1
        first_element = self._generate_tensor(shape, dtype)
        if dtype == "bool":
            scalar = True
        else:
            scalar = 0b1100
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, scalar)
```
