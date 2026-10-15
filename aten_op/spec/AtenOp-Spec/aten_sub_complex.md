# aten.sub.complex

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.sub.complex(complex a, complex b) -> complex
```
### 1.2 算子功能
`aten.sub.complex` 是纯标量重载：对两个复数标量执行减法 out = a - b，按复数相减（实部减实部、虚部减虚部），返回 complex。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `complex` | 被减数（复数标量） |
| `b` | `complex` | 减数（复数标量） |
| 返回值 | `complex` | a - b 的结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | - | - | - | - | √ | √ |
| NPU | - | - | - | - | - | - | - | √ | √ |

> 纯标量重载：`a`/`b` 固定为 complex，返回 complex，与 tensor dtype/shape 无关，不参与 tensor 的 dtype 判定。

#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

> 纯标量重载在 `torch.compile(inductor)` 下被常量折叠，不会生成 kernel。


### 2.4 使用方法


```python
    dtypes = ['complex64', 'complex128']

    @parametrize('dtype', dtypes)
    def test_aten_sub_complex(self, dtype):
        def op_calc(first_element, first_scalar, second_scalar):
            y = torch.ops.aten.sub.complex(first_scalar, second_scalar)
            result = first_element + y
            return result

        def op_calc_1(first_element, first_scalar, second_scalar):
            y = torch.ops.aten.sub.complex(first_scalar, second_scalar)
            result = first_element + y + y
            return result

        zero_tensor = torch.zeros(64, dtype=torch.complex64, device=torch.device("npu"))
        first_scalar = torch.tensor(complex(3.0, 0.5), dtype=getattr(torch, dtype)).item()
        second_scalar = torch.tensor(complex(1.0, 0.25), dtype=getattr(torch, dtype)).item()
        self.execute_inductor_test((64,), dtype, op_calc, op_calc_1, zero_tensor, first_scalar, second_scalar)
```
