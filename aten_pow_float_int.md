# aten.pow.float_int

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.pow.float_int(float a, int b) -> float
```

### 1.2 算子功能

计算浮点标量底数 `a` 的整数次幂 `a ** b`，返回浮点标量。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `float` | 幂运算的底数。 |
| `b` | `int` | 幂运算的整数指数。 |
| `返回值` | `float` | 标量幂运算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持


|  | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | √ | √ | √ | - | - | - |
| NPU | - | - | - | √ | √ | √ | - | - | - |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

### 2.3 特殊限制说明

无

### 2.4 使用方法

```python
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('dtype', dtypes)
    def test_aten_pow_float_int(self, dtype):
        def op_calc(first_element, scalar_a, scalar_b):
            y = torch.ops.aten.pow.float_int(scalar_a, scalar_b)
            return first_element + y

        def op_calc_1(first_element, scalar_a, scalar_b):
            y = torch.ops.aten.pow.float_int(scalar_a, scalar_b)
            return first_element + y + y

        first_element = torch.zeros(64, dtype=getattr(torch, dtype), device="npu")
        scalar_a = self._generate_safe_pow_float(dtype)
        scalar_b = self._generate_safe_pow_int(dtype)
        self.execute_inductor_test((64,), dtype, op_calc, op_calc_1,
                                   first_element, scalar_a, scalar_b)
```
