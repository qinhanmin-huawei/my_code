# aten.mul.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.mul.default(Scalar a, Scalar b) -> Scalar
```

### 1.2 算子功能
`aten.mul.default` 对两个标量执行乘法，返回标量结果。输入可以是整数、浮点数、布尔值或复数，结果遵循 PyTorch 类型提升规则。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `a` | `Scalar` | 第一个输入标量 |
| `b` | `Scalar` | 第二个输入标量 |
| 返回值 | `Scalar` | 两个输入标量的乘积 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| ------ | ---- | ---- |  ----  |  ---- |  ----  | ----- | ----- | ----- | ----- | ------ | ------ | ------ | --------- | ---------- |
| GPU    | -    | √    | -      | -     | -      | √     | -     | √     | √     | √      | √      | √      | √         | √         |
| NPU    | -    | √    | -      | -     | -      | √     | -     | √     | √     | √      | √      | √      | -         | -          |

#### 2.2.2 Shape 支持
1、仅支持零维标量输入，不涉及 shape 泛化。

2、输入参数 `a`、`b` 支持 `int8`、`int32`、`int64`、`float16`、`float32`、`bfloat16`、`bool`、`complex64`；`complex128` 不支持。

### 2.3 特殊限制说明
> 本文仅覆盖 `aten.mul.default` 重载。固定标量可能在编译阶段常量折叠，因此暂不做性能和 lowering 验证，仅维护规格。

### 2.4 使用方法
```python
    def test_aten_mul_default(self):
        def op_calc(anchor, first_scalar, second_scalar):
            scalar_result = torch.ops.aten.mul.default(first_scalar, second_scalar)
            return anchor + scalar_result

        def op_calc_1(anchor, first_scalar, second_scalar):
            scalar_result = torch.ops.aten.mul.default(first_scalar, second_scalar)
            return anchor + scalar_result + scalar_result

        anchor = torch.zeros((128,), dtype=torch.float32, device='npu')
        first_scalar, second_scalar = 2.0, 3.0
        self.execute_inductor_test((128,), 'float32', op_calc, op_calc_1,
                                   anchor, first_scalar, second_scalar,
                                   op_name='aten.add')
```