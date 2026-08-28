# aten.bitwise_or.Scalar_Tensor_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.bitwise_or.Scalar_Tensor_out(self: Scalar, other: Tensor, *, out: Tensor(a!)) -> Tensor(a!)
```
### 1.2 算子功能
`aten.bitwise_or.Scalar_Tensor_out` 计算标量值与输入张量之间的**按位或（Bitwise OR）** 运算，并将结果保存到指定的 `out` 参数中。

- **整型按位或**：对标量整数的二进制位与张量中每个整数的二进制位逐位进行或运算。规则为：只要有一个对应位为 `1`，结果位即为 `1`；否则为 `0`。
- **布尔型逻辑或**：如果输入张量是布尔型，算子执行逻辑或（Logical OR）运算。
- **输出重载特性**：该变体不会分配新的张量，而是直接修改传入的 `out` 张量。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self` | `Scalar`（`int` / `bool`） | 用于按位或运算的标量值 |
| `other` | `Tensor` | 输入的源张量，须为整型或布尔类型 |
| `out` | `Tensor(a!)` | 用于保存输出结果的张量，形状须与输入一致，数据类型须与输入一致。该张量会被原地修改 |
| 返回值 | `Tensor(a!)` | 返回 `out` 张量本身（即计算结果） |

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

    def test_aten_bitwise_or_Scalar_Tensor_out(self, shape, dtype):
        def op_calc(tensor, out_t):
            y = torch.ops.aten.bitwise_or.Scalar_Tensor_out(6, tensor, out = out_t)
            return y
        def op_calc_1(tensor, out_t):
            y = torch.ops.aten.bitwise_or.Scalar_Tensor_out(6, tensor, out = out_t)
            return y + y
        tensor = self._generate_tensor(shape, dtype)
        if dtype == "bool":
            out = torch.empty(shape,dtype=torch.int64, device='npu')
        else:
            out = torch.empty(shape,dtype = eval("torch."+ dtype), device='npu')
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, out)

```
