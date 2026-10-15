# aten.__ior__

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.__ior__(self: Tensor(a!), other: Scalar) -> Tensor(a!)
```
### 1.2 算子功能
`aten.__ior__` 是 Python 运算符重载接口（对应 `|=`），与 `aten.bitwise_or_` 等价，是 **原地（in-place）版本** 的**按位或（Bitwise OR）** 运算。它计算输入张量 `self` 与标量 `other` 之间的按位或结果，并**直接写回**调用张量（`self`）中。

- **整型按位或**：对张量中每个元素的二进制位与标量 `other` 的二进制位进行逐位或运算。规则为：两个对应位中只要有一个为 `1`，结果位即为 `1`；否则为 `0`。
- **布尔型逻辑或**：如果输入是布尔张量，算子执行逻辑或（Logical OR）运算。
- **原地操作特性**：以 `__ior__` 命名，表示**原地修改**，不会分配新的张量，直接修改原始张量数据。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor(a!)` | 输入张量，**会被原地修改**。须为整型或布尔类型 |
| `other` | `Scalar` | 用于按位或运算的标量值（Python 整数或布尔值），须与 `self` 类型兼容 |
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

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten___ior__(self, shape, dtype):
        def op_calc(tensor, tensor2):
            y = torch.ops.aten.__ior__(tensor, tensor2)
            return y
        def op_calc_1(tensor, tensor2):
            y = torch.ops.aten.__ior__(tensor, tensor2)
            return y
        tensor = self._generate_tensor(shape, dtype)
        tensor2 = torch.full(shape, 2, dtype=tensor.dtype, device = "npu")
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, tensor2)
```
