# aten.true_divide.Scalar

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.true_divide.Scalar(self: Tensor, other: Scalar) -> Tensor
```

### 1.2 算子功能

将张量逐元素除以标量，始终使用真除法语义。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 被除数张量。 |
| `other` | `Scalar` | 非零除数。 |
| `返回值` | `Tensor` | 逐元素真除法结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |

整数和 bool 输入会提升为默认浮点类型。

#### 2.2.2 Shape 支持

输入 Tensor 支持 1～4 维。


### 2.3 特殊限制说明

测试使用非零标量除数。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'float32', 'bfloat16']
    shapes = [(32,), (32, 32), (32, 32, 32), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_true_divide_scalar(self, shape, dtype):
        def op_calc(first_element, other):
            x = first_element
            y = torch.ops.aten.true_divide.Scalar(x, other)
            result = y
            return result

        def op_calc_1(first_element, other):
            x = first_element
            y = torch.ops.aten.true_divide.Scalar(x, other)
            result = y + y
            return result

        tensor1 = self._generate_tensor(shape, dtype)
        other = 3
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1, other)
```
