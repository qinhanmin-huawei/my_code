# aten.mul.complex_float

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.mul.complex_float(complex a, float b) -> complex
```

### 1.2 算子功能
`aten.mul.complex_float` 对一个复数标量和一个浮点标量执行乘法，返回复数标量结果。重载名中的 `float` 表示第二个操作数是实数浮点标量，并不表示复数的实部和虚部采用某种单独的编码格式。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `a` | `complex` | 复数标量 |
| `b` | `float` | 实数浮点标量 |
| 返回值 | `complex` | 复数与浮点标量的乘积 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| ------ | ---- | ---- |  ----  |  ---- |  ----  | ----- | ----- | ----- | ----- | ------ | ------ | ------ | --------- | ---------- |
| GPU    | -    | ×    | -      | -     | -      | ×     | -     | ×     | √     | √      | √      | ×      | √         | √          |
| NPU    | -    | ×    | -      | -     | -      | ×     | -     | ×     | √     | √      | √      | ×      | √         | ×          |

#### 2.2.2 Shape 支持
1、`mul.complex_float` 本身仅处理零维复数/浮点标量，不涉及张量 shape。

2、输入参数 `a` 支持 `complex64`，参数 `b` 支持 `float16`、`float32`、`bfloat16`；`complex128` 不支持。

### 2.3 特殊限制说明
> 标量算子不涉及 shape 泛化。该算子会在编译阶段对固定标量进行常量折叠，因此暂不做性能和 lowering 验证，仅维护规格。复数操作数应使用统一精度；`b` 为浮点标量。复数与浮点输入的支持类型按并集标记。

### 2.4 使用方法
```python
    def test_aten_mul_complex_float(self):
        def op_calc(anchor, complex_value, float_value):
            scalar_result = torch.ops.aten.mul.complex_float(complex_value, float_value)
            return anchor + scalar_result

        def op_calc_1(anchor, complex_value, float_value):
            scalar_result = torch.ops.aten.mul.complex_float(complex_value, float_value)
            return anchor + scalar_result + scalar_result

        anchor = torch.zeros((128,), dtype=torch.complex64, device='npu')
        complex_value, float_value = complex(1.0, 2.0), 3.0
        self.execute_inductor_test((128,), 'complex64', op_calc, op_calc_1,
                                   anchor, complex_value, float_value,
                                   op_name='aten.add')
```