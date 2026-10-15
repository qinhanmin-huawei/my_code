# aten.remainder.Scalar

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.remainder.Scalar(self: Tensor, other: Scalar) -> Tensor
```


### 1.2 算子功能
`aten.remainder.Scalar` 计算输入张量 `self` 中每个元素与标量 `other` 的**逐元素取余（remainder）** 运算，并返回新的张量。

- 对于整数张量：执行整数取余，结果符号与被除数（即 `self` 的元素）相同。
- 对于浮点张量：执行浮点取余，结果为 `self - other * floor(self / other)`，符号与被除数相同。
- 标量 `other` 会被转换为与张量元素相同的类型，然后逐元素计算。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 输入张量（被除数），支持浮点或整型 |
| `other` | `Scalar`（`int` / `float`） | 除数标量，须可转换为张量元素类型，且不能为零（整数零会报错，浮点零产生 inf 或 nan） |
| 返回值 | `Tensor` | 新的张量，形状与 `self` 相同，包含逐元素取余的结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64','float16', 'bfloat16', 'float32', "bool"]
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]
    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_remainder_scalar(self, shape, dtype):
        def op_calc(first_element):
            result = torch.ops.aten.remainder.Scalar(first_element, 1.5)
            return result
        def op_calc_1(first_element):
            result = torch.ops.aten.remainder.Scalar(first_element, 1.5)
            return result + result
        first_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)
```
