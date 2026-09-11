# aten.pow.int_float

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.pow.int_float(int a, float b) -> float
```

### 1.2 算子功能

计算整数标量底数 `a` 的浮点次幂 `a ** b`，返回浮点标量。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `int` | 幂运算的底数。 |
| `b` | `float` | 幂运算的浮点指数。 |
| `返回值` | `float` | 标量幂运算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|  | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | √ | √ | √ | - | - | - | - | - | - |
| NPU | √ | √ | √ | - | - | - | - | - | - |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

### 2.3 特殊限制说明

无

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64']

    @parametrize('dtype', dtypes)
    def test_aten_pow_int_float(self, dtype):
        def op_calc(first_element, scalar_a, scalar_b):
            y = torch.ops.aten.pow.int_float(scalar_a, scalar_b)
            return first_element + y

        def op_calc_1(first_element, scalar_a, scalar_b):
            y = torch.ops.aten.pow.int_float(scalar_a, scalar_b)
            return first_element + y + y

        first_element = torch.zeros(64, dtype=getattr(torch, dtype), device="npu")
        scalar_a = self._generate_safe_pow_int(dtype)
        scalar_b = self._generate_safe_pow_fractional_exponent(dtype)
        self.execute_inductor_test((64,), dtype, op_calc, op_calc_1,
                                   first_element, scalar_a, scalar_b)
```
