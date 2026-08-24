# aten.isinf

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.isinf(Tensor self) -> Tensor
```
### 1.2 算子功能
`aten.isinf` 的核心逻辑是对输入张量中的每个元素进行检查，判断其是否为无穷大（正无穷或负无穷），并返回一个与输入张量形状相同的布尔类型张量：

1. **逐元素判断**：遍历输入张量（`self`）的每一个元素，若该元素的值为正无穷（`+inf`）或负无穷（`-inf`），则输出对应的 `True`；否则输出 `False`。
2. **NaN 处理**：NaN（非数值）不属于无穷大，对应位置输出 `False`。
3. **整数类型处理**：对于整数类型的输入张量，由于整数类型无法表示无穷大，输出张量的所有元素均为 `False`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量，支持浮点类型和整数类型 |
| 返回值 | `Tensor` | 与 `self` 形状相同的布尔类型张量，元素位置为无穷大时取 `True`，否则取 `False` |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ | × |
| NPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、输出张量的形状与输入张量相同

### 2.3 特殊限制说明

> 无

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64','float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_isinf(self, shape, dtype):
        def op_calc(input):
            y = torch.ops.aten.isinf(input)
            return y
        def op_calc_1(input):
            y = torch.ops.aten.isinf(input)
            # isinf 返回 bool 张量，用逻辑或消费 y 两次以触发融合
            return y | y
        input = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input)
```
