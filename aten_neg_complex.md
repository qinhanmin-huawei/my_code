# aten.neg.complex

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.neg.complex(complex a) -> complex
```

### 1.2 算子功能
`aten.neg.complex` 对复数标量执行取负，实部和虚部同时取相反数。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `a` | `complex` | 待取负的复数标量 |
| 返回值 | `complex` | 输入复数标量的相反数 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| ------ | ---- | ---- |  ----  |  ---- |  ----  | ----- | ----- | ----- | ----- | ------ | ------ | ------ | --------- | ---------- |
| GPU    | -    | ×    | -      | -     | -      | ×     | -     | ×     | ×     | ×      | ×      | ×      | √         | √          |
| NPU    | -    | ×    | -      | -     | -      | ×     | -     | ×     | ×     | ×      | ×      | ×      | √         | ×          |

#### 2.2.2 Shape 支持
1、仅支持零维复数标量输入，不涉及 shape 泛化。

2、输入参数 `a` 支持 `complex64`；`complex128` 不支持。

### 2.3 特殊限制说明
> 标量结果可能在编译阶段常量折叠；用例通过零值 Tensor 承接结果验证 eager/Inductor 语义，允许 output code 中不保留 `neg.complex`。

### 2.4 使用方法
```python
    def test_aten_neg_complex(self):
        def op_calc(anchor, scalar):
            scalar_result = torch.ops.aten.neg.complex(scalar)
            return anchor + scalar_result

        def op_calc_1(anchor, scalar):
            scalar_result = torch.ops.aten.neg.complex(scalar)
            return anchor + scalar_result + scalar_result

        anchor = torch.zeros((128,), dtype=torch.complex64, device='npu')
        self.execute_inductor_test((128,), 'complex64', op_calc, op_calc_1,
                                   anchor, complex(1.0, 2.0),
                                   op_name='aten.add')
```