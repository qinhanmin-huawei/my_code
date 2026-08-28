# aten.any.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.any.default(Tensor self) -> Tensor
```
### 1.2 算子功能
`aten.any.default` 是全局归约算子，判断输入张量中是否**存在至少一个非零（True）元素**，返回一个 0 维布尔标量张量：

1. **归约判定**：遍历 `self` 全部元素，只要有一个元素的值非零（浮点非零、整数非零、bool 为 True），结果即为 `True`。
2. **全零情况**：所有元素均为零时返回 `False`。
3. **NaN 处理**：NaN 不等于零，视为"非零"，参与归约后使结果为 `True`。
4. **overload 关系**：`.default` 为无额外参数的显式 overload，行为与 `aten.any` 完全一致。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| 返回值 | `Tensor` | bool 类型 0 维标量张量，存在非零元素为 `True`，否则为 `False` |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ |  |
| NPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ |  |

#### 2.2.2 Shape 支持

1、shape 维度范围：[1-4]

2、输出为 0 维标量张量，与输入形状无关

### 2.3 特殊限制说明

> 无

### 2.4 使用方法

```python
    dtypes = ['int8','int32','int64','float16', 'bfloat16', 'float32']  
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_aten_any_default(self, shape, dtype):
        def op_calc(first_element):
            y = torch.ops.aten.any.default(first_element)
            return y

        def op_calc_1(first_element):
            y = torch.ops.aten.any.default(first_element)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)
```
