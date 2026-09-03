# aten.bitwise_right_shift.Scalar_Tensor_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.bitwise_right_shift.Scalar_Tensor_out(self: Scalar, other: Tensor, *, out: Tensor) -> Tensor
```
### 1.2 算子功能
`aten.bitwise_right_shift.Scalar_Tensor_out` 对标量 `self` 进行按位右移，移动位数由张量 `other` 中每个元素指定，并将结果写入 `out` 张量。

- 将标量 `self` 广播（broadcast）到与张量 `other` 相同的形状。
- 对广播后的标量逐元素与 `other` 的对应元素执行按位右移运算。
- 将结果存入 `out` 张量（形状与 `other` 相同）。
- **输出重载特性**：不会分配新张量，直接修改 `out`，减少内存开销。

- **整型按位右移**：对标量（整数）的二进制位向右移动 `other` 对应元素指定的位数。
  - 若有符号标量（如 Python `int`），执行算术右移（符号位扩展）。
  - 若无符号标量（如 `uint8`），执行逻辑右移（高位补0）。
- **布尔型**：若标量为 `bool` 或张量为布尔类型，执行逻辑右移（`True`/`False` 视为 1/0，结果仍为整数）。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self` | `Scalar`（`int` / `bool`） | 被右移的标量值，须为整数或布尔值 |
| `other` | `Tensor` | 张量，包含每个元素的右移位数，须为整型或布尔类型 |
| `out` | `Tensor` | 用于保存输出结果的张量，形状须与 `other` 一致，数据类型须与结果兼容，该张量会被原地修改 |
| 返回值 | `Tensor` | 返回 `out` 张量本身（即计算结果） |


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
    def test_aten_bitwise_right_shift_Scalar_Tensor_out(self, shape, dtype):
        def op_calc(first_element, out):
            torch.ops.aten.bitwise_right_shift.Scalar_Tensor_out(0b1100, first_element, out=out)
            return out
        def op_calc_1(first_element, out):
            torch.ops.aten.bitwise_right_shift.Scalar_Tensor_out(0b1100, first_element, out=out)
            return out + out
        if dtype == "bool":
            out = torch.empty(shape,dtype=torch.int64, device='npu')
        else:
            out = torch.empty(shape,dtype = eval("torch."+ dtype), device='npu')
        first_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, out)
```
