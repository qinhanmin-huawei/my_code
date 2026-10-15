# aten.pow.Scalar_Scalar

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.pow.Scalar_Scalar(Scalar self, Scalar exponent) -> Scalar
```

### 1.2 算子功能
对两个标量执行幂运算，返回标量结果。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Scalar` | 标量底数 |
| `exponent` | `Scalar` | 标量指数 |
| 返回值 | `Scalar` | 幂运算结果 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × | - | - |

#### 2.2.2 Shape 支持
仅支持零维标量输入，不涉及 shape 泛化。

### 2.3 特殊限制说明
> 纯标量结果可能在编译阶段常量折叠；用例通过零值 Tensor 承接结果验证 eager/Inductor 语义。`bool` 不支持。

### 2.4 使用方法
```python
    def test_aten_pow_Scalar_Scalar(self):
        def op_calc(anchor, base, exponent):
            result = torch.ops.aten.pow.Scalar_Scalar(base, exponent)
            return anchor + result

        def op_calc_1(anchor, base, exponent):
            result = torch.ops.aten.pow.Scalar_Scalar(base, exponent)
            return anchor + result + result

        anchor = torch.zeros((128,), dtype=torch.float32, device='npu')
        self.execute_inductor_test((128,), 'float32', op_calc, op_calc_1,
                                   anchor, 2.0, 3.0, op_name='aten.add')
```