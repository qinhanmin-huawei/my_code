# aten.mul.float_int

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.mul.float_int(float a, int b) -> float
```

### 1.2 算子功能
`aten.mul.float_int` 对一个浮点标量和一个整数标量执行乘法，返回浮点标量结果。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `a` | `float` | 浮点标量 |
| `b` | `int` | 整数标量 |
| 返回值 | `float` | 浮点标量与整数标量的乘积 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| ------ | ---- | ---- |  ----  |  ---- |  ----  | ----- | ----- | ----- | ----- | ------ | ------ | ------ | --------- | ---------- |
| GPU    | -    | √    | -      | -     | -      | √     | -     | √     | √     | √      | √      | ×      | ×         | ×          |
| NPU    | -    | √    | -      | -     | -      | √     | -     | √     | √     | √      | √      | ×      | ×         | ×          |

#### 2.2.2 Shape 支持
1、仅支持零维浮点/整数标量输入，不涉及 shape 泛化。

2、输入参数 `a` 支持 `float16`、`float32`、`bfloat16`，参数 `b` 支持 `int8`、`int32`、`int64`。

### 2.3 特殊限制说明
> 本文仅覆盖 `aten.mul.float_int` 重载。固定标量可能在编译阶段常量折叠，因此暂不做性能和 lowering 验证，仅维护规格。浮点与整数输入的支持类型按并集标记。

### 2.4 使用方法
```python
    def test_aten_mul_float_int(self):
        def op_calc(anchor, float_value, int_value):
            scalar_result = torch.ops.aten.mul.float_int(float_value, int_value)
            return anchor + scalar_result

        def op_calc_1(anchor, float_value, int_value):
            scalar_result = torch.ops.aten.mul.float_int(float_value, int_value)
            return anchor + scalar_result + scalar_result

        anchor = torch.zeros((128,), dtype=torch.float32, device='npu')
        float_value, int_value = 2.5, 3
        self.execute_inductor_test((128,), 'float32', op_calc, op_calc_1,
                                   anchor, float_value, int_value,
                                   op_name='aten.add')
```