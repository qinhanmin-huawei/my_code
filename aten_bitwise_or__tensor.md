# aten.bitwise_or_.Tensor

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.bitwise_or_.Tensor(self: Tensor(a!), other: Tensor) -> Tensor(a!)
```
### 1.2 算子功能
`aten.bitwise_or_.Tensor` 是 `aten.bitwise_or` 的**原地（in-place）版本**，专门针对 `other` 参数为张量（Tensor）的情况。它计算输入张量 `self` 与另一个张量 `other` 之间的**按位或（Bitwise OR）** 运算，并将结果**直接写回** `self` 中。

- **整型按位或**：对两个张量对应元素的二进制位进行逐位或运算。规则：两个对应位中只要有一个为 `1`，结果位即为 `1`；否则为 `0`。
- **布尔型逻辑或**：如果输入是布尔张量，算子执行逻辑或（Logical OR）运算。
- **原地操作特性**：函数名以下划线 `_` 结尾，且后缀 `.Tensor` 表明 `other` 为张量类型，该版本不会分配新的张量，直接修改原始张量数据。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor(a!)` | 输入张量，**会被原地修改**。须为整型或布尔类型 |
| `other` | `Tensor` | 另一个输入张量，须与 `self` 可广播（broadcastable）且类型兼容 |
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
    def test_aten_bitwise_or__tensor(self, shape, dtype):
        def op_calc(first_element, second_element):
            torch.ops.aten.bitwise_or_.Tensor(first_element, second_element)
            return first_element

        def op_calc_1(first_element, second_element):
            torch.ops.aten.bitwise_or_.Tensor(first_element, second_element)
            return first_element + first_element

        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, second_element)
```
