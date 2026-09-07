# aten.mul.int_complex

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.mul.int_complex(int a, complex b) -> complex
```

### 1.2 算子功能
`aten.mul.int_complex` 对一个整数标量和一个复数标量执行乘法，返回复数标量结果。复数的实部和虚部应使用兼容精度。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `a` | `int` | 整数标量 |
| `b` | `complex` | 复数标量 |
| 返回值 | `complex` | 整数标量与复数标量的乘积 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| ------ | ---- | ---- |  ----  |  ---- |  ----  | ----- | ----- | ----- | ----- | ------ | ------ | ------ | --------- | ---------- |
| GPU    | -    | √    | -      | -     | -      | √     | -     | √     | ×     | ×      | ×      | ×      | √         | ×          |
| NPU    | -    | √    | -      | -     | -      | √     | -     | √     | ×     | ×      | ×      | ×      | √         | ×          |

#### 2.2.2 Shape 支持
1、仅支持零维整数/复数标量输入，不涉及 shape 泛化。

2、输入参数 `a` 支持 `int8`、`int32`、`int64`，参数 `b` 支持 `complex64`；`complex128` 不支持。

### 2.3 特殊限制说明
> 固定标量可能在编译阶段常量折叠，因此暂不做性能和 lowering 验证，仅维护规格。复数与整数输入的支持类型按并集标记。

### 2.4 使用方法
```python
    def test_aten_mul_int_complex(self):
        def op_calc(anchor, int_value, complex_value):
            scalar_result = torch.ops.aten.mul.int_complex(int_value, complex_value)
            return anchor + scalar_result

        def op_calc_1(anchor, int_value, complex_value):
            scalar_result = torch.ops.aten.mul.int_complex(int_value, complex_value)
            return anchor + scalar_result + scalar_result

        anchor = torch.zeros((128,), dtype=torch.complex64, device='npu')
        int_value, complex_value = 3, complex(1.0, 2.0)
        self.execute_inductor_test((128,), 'complex64', op_calc, op_calc_1,
                                   anchor, int_value, complex_value,
                                   op_name='aten.add')
```