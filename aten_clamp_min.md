# aten.clamp_min

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.clamp_min(self: Tensor, min: Scalar) -> Tensor
```

### 1.2 算子功能

逐元素计算 `max(self, min)`，将低于下界的元素截断到 `min`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 源张量。 |
| `min` | `Scalar` | 逐元素截断使用的下界。 |
| `返回值` | `Tensor` | 与输入 shape 相同的新张量。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | x |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | x |



#### 2.2.2 Shape 支持

1. 本测试覆盖的 shape 维度范围为 `[1, 4]`。
2. 测试 shape：`(128,)`、`(128, 128)`、`(128, 1024, 4096)`、`(4, 8, 16, 32)`。


### 2.3 特殊限制说明

输入与下界必须位于兼容设备；不支持复数 Tensor。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'float32', 'bfloat16']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_clamp_min(self, shape, dtype):
        def op_calc(first_element, min_value):
            x = first_element
            y = torch.ops.aten.clamp_min(x, min_value)
            result = y
            return result

        def op_calc_1(first_element, min_value):
            x = first_element
            y = torch.ops.aten.clamp_min(x, min_value)
            result = y + y
            return result

        tensor1 = self._generate_tensor(shape, dtype)
        min_value = self._generate_random_scalar(dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1, min_value)
```
