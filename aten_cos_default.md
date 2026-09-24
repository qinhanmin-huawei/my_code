# aten.cos.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.cos.default(self: Tensor) -> Tensor
```

### 1.2 算子功能

逐元素计算输入的余弦值。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 源张量。 |
| `返回值` | `Tensor` | 逐元素计算后的张量。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |


#### 2.2.2 Shape 支持

输入 Tensor 支持 1～4 维。


### 2.3 特殊限制说明

无。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'float32', 'bfloat16']
    shapes = [(32,), (32, 32), (32, 32, 32), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_cos_default(self, shape, dtype):
        def op_calc(first_element):
            x = first_element
            y = torch.ops.aten.cos.default(x)
            result = y
            return result

        def op_calc_1(first_element):
            x = first_element
            y = torch.ops.aten.cos.default(x)
            result = y + y
            return result

        tensor1 = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1)
```
