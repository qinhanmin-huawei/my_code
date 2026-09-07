# aten.remainder.Scalar_Tensor

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.remainder.Scalar_Tensor(self: Scalar, other: Tensor) -> Tensor
```


### 1.2 算子功能
`aten.remainder.Scalar_Tensor` 对标量 `self` 与张量 `other` 中每个元素进行**逐元素取余（remainder）** 运算，并返回新的张量。

- 标量 `self` 作为被除数，张量 `other` 中的每个元素作为除数。
- 对于整数类型：执行整数取余，结果符号与被除数（即标量 `self` 的符号）相同。
- 对于浮点类型：执行浮点取余，结果为 `self - other * floor(self / other)`，符号与被除数相同。
- 标量 `self` 会被广播（broadcast）到与 `other` 相同的形状。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self` | `Scalar`（`int` / `float`） | 被除数标量，须可转换为张量元素类型 |
| `other` | `Tensor` | 除数张量，支持浮点或整型，元素不能为零（整数零会报错，浮点零产生 inf 或 nan） |
| 返回值 | `Tensor` | 新的张量，形状与 `other` 相同，包含逐元素取余的结果 |

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
    def test_aten_remainder_scalar_tensor(self, shape, dtype):
        def op_calc(first_element):
            y = torch.ops.aten.remainder.Scalar_Tensor(1.5, first_element)
            result = y
            return result
        def op_calc_1(first_element):
            y = torch.ops.aten.remainder.Scalar_Tensor(1.5, first_element)
            result = y + 1
            return result

        first_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)
```
