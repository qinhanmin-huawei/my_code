# aten.mul.Tensor

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.mul(Tensor self, Tensor other) -> Tensor
```

### 1.2 算子功能
`aten.mul` 对两个输入张量执行逐元素乘法（Element-wise multiplication）。输入张量的形状须相同，或满足 PyTorch 的广播（broadcasting）规则；输出张量为广播后的形状。

- 对整数和浮点输入，输出的每个元素等于对应输入元素的乘积。
- 对布尔输入，算子等价于逐元素逻辑与（`True * True` 为 `True`，其他情况为 `False`）。
- 输入数据类型不同时，按 PyTorch 类型提升规则确定计算和返回类型；同一测试用例中的输入建议使用相同数据类型。
- 该重载不修改输入张量，会返回新分配的张量。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 参与乘法运算的第一个源张量 |
| `other` | `Tensor` | 参与乘法运算的第二个源张量，须与 `self` 形状相同或可广播 |
| 返回值 | `Tensor` | 广播后形状的结果张量，数据类型遵循类型提升规则 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | ----- | ----- | ----- | ------ | ------ | ------ |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、输入张量维度范围：[1-4]。

2、两个输入张量的形状须相同或满足广播规则；输出形状为各维度广播后的结果形状。
### 2.3 特殊限制说明

> 本文仅覆盖 `aten.mul.Tensor` 重载

### 2.4 使用方法

```python
    dtypes = ['uint8', 'int8', 'int16', 'int32', 'int64',
              'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_mul(self, shape, dtype):
        def op_calc(first_element, second_element):
            y = torch.ops.aten.mul(first_element, second_element)
            return y

        def op_calc_1(first_element, second_element):
            y = torch.ops.aten.mul(first_element, second_element)
            return y + y

        tensor1 = self._generate_tensor(shape, dtype)
        tensor2 = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tensor1, tensor2)
```