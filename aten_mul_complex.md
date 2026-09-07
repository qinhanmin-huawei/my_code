# aten.mul.complex

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.mul.complex(complex a, complex b) -> complex
```

### 1.2 算子功能
`aten.mul.complex` 计算两个复数标量的乘积并返回复数标量结果。输入可由 Python `complex` 或标准复数 Tensor 的零维元素表示。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `a` | `complex` | 第一个复数标量 |
| `b` | `complex` | 第二个复数标量 |
| 返回值 | `complex` | 两个输入标量的复数乘积 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| ------ | ---- | ---- |  ----  |  ---- |  ----  | ----- | ----- | ----- | ----- | ------ | ------ | ------ | --------- | ---------- |
| GPU    | -    | ×    | -      | -     | -      | ×     | -     | ×     | ×     | ×      | ×      | ×      | √         | ×          |
| NPU    | -    | ×    | -      | -     | -      | ×     | -     | ×     | ×     | ×      | ×      | ×      | √         | ×          |

#### 2.2.2 Shape 支持
1、`mul.complex` 本身仅处理零维复数标量，不涉及张量 shape。

2、输入参数 `a`、`b` 均支持 `complex64`；`complex128` 不支持。

### 2.3 特殊限制说明
> 本文仅覆盖 `aten.mul.complex` 重载。标量算子不涉及 shape 泛化。该算子会在编译阶段对固定标量进行常量折叠，因此暂不做性能和 lowering 验证，仅维护规格。复数的实部和虚部应使用同一精度：`complex64` 对应 `float32` 分量，`complex128` 对应 `float64` 分量。

### 2.4 使用方法
```python
    def test_aten_mul_complex(self):
        def op_calc(anchor, first_complex, second_complex):
            scalar_result = torch.ops.aten.mul.complex(first_complex, second_complex)
            return anchor + scalar_result

        def op_calc_1(anchor, first_complex, second_complex):
            scalar_result = torch.ops.aten.mul.complex(first_complex, second_complex)
            return anchor + scalar_result + scalar_result

        anchor = torch.zeros((128,), dtype=torch.complex64, device='npu')
        first_complex, second_complex = complex(1.0, 2.0), complex(3.0, 4.0)
        self.execute_inductor_test((128,), 'complex64', op_calc, op_calc_1,
                                   anchor, first_complex, second_complex,
                                   op_name='aten.add')
```