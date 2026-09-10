# aten.div.float

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.div.float(float a, float b) -> float
```
### 1.2 算子功能
`aten.div.float` 是纯标量重载：对两个 float 标量执行真除法 a / b，返回 float。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `float` | 被除数标量 |
| `b` | `float` | 除数标量，为 0 时按浮点语义得 inf/nan |
| 返回值 | `float` | a / b 的结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | √ | √ | √ | - | - | - |
| NPU | - | - | - | √ | √ | √ | - | - | - |


#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

无

### 2.4 使用方法


```python
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('dtype', dtypes)
    def test_aten_div_float(self, shape, dtype):
        def op_calc(first_element, scalar_a, scalar_b):
            y = torch.ops.aten.div.float(scalar_a, scalar_b)
            result = first_element + y
            return result

        def op_calc_1(first_element, scalar_a, scalar_b):
            y = torch.ops.aten.div.float(scalar_a, scalar_b)
            result = first_element + y + y
            return result

        first_element = torch.zeros(shape, dtype=getattr(torch, dtype), device="npu")
        scalar_a = self._generate_random_scalar(dtype)
        scalar_b = self._generate_random_scalar(dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, scalar_a, scalar_b)
```
