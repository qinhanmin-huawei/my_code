# aten.div

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.div(self: Tensor, other: Tensor) -> Tensor
```

### 1.2 算子功能

对两个张量逐元素执行除法；该 packet 调用在两个 Tensor 参数下选择 Tensor overload。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 被除数。 |
| `other` | `Tensor` | 除数。 |
| `返回值` | `Tensor` | 广播后的逐元素商。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

未指定 `rounding_mode` 时，整数和 bool 输入会提升为默认浮点类型。

#### 2.2.2 Shape 支持

输入 Tensor 支持 1～4 维。

### 2.3 特殊限制说明

测试数据会排除零除数以避免 `inf`/`nan` 干扰精度校验。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool']
    shapes = [(32,), (32, 32), (32, 32, 32), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_div(self, shape, dtype):
        def op_calc(first_element, second_element):
            x = first_element
            y = torch.ops.aten.div(x, second_element)
            result = y
            return result

        def op_calc_1(first_element, second_element):
            x = first_element
            y = torch.ops.aten.div(x, second_element)
            result = y + y
            return result

        tensor1 = self._generate_tensor(shape, dtype)
        tensor2 = self._generate_tensor(shape, dtype, min=1, max=10)
        if dtype == 'bool':
            tensor2 = torch.ones_like(tensor2)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1, tensor2)
```
