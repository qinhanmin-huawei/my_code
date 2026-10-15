# aten.exp.complex

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.exp.complex(complex a) -> complex
```
### 1.2 算子功能
`aten.exp.complex` 是纯标量重载：对复数标量 a 计算自然指数 e^a，返回 complex（按欧拉公式 e^(x+iy) = e^x × (cos y + i sin y)）。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `complex` | 输入复数标量 |
| 返回值 | `complex` | e^a 的结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | - | - | - | - | √ | √ |
| NPU | - | - | - | - | - | - | - | √ | √ |

> 纯标量重载：`a` 固定为 complex，返回 complex，与 tensor dtype/shape 无关，不参与 tensor 的 dtype 判定。

#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

> 纯标量重载在 `torch.compile(inductor)` 下被常量折叠，不会生成 kernel，测试记录 `CAN NOT lowering` 为预期行为。
> 输入标量按 complex64/complex128 量化后再喂入，区分单双精度；载体 tensor 用固定 complex64，仅作融合触发。


### 2.4 使用方法


```python
    dtypes = ['complex64', 'complex128']

    @parametrize('dtype', dtypes)
    def test_aten_exp_complex(self, dtype):
        # aten::exp.complex(complex a) -> complex
        def op_calc(first_element, scalar):
            y = torch.ops.aten.exp.complex(scalar * 1e-3)
            result = first_element + y
            return result

        def op_calc_1(first_element, scalar):
            y = torch.ops.aten.exp.complex(scalar * 1e-3)
            result = first_element + y + y
            return result

        zero_tensor = torch.zeros(64, dtype=torch.complex64, device=torch.device("npu"))
        scalar = torch.tensor(complex(0.5, 0.25), dtype=getattr(torch, dtype)).item()
        self.execute_inductor_test((64,), dtype, op_calc, op_calc_1, zero_tensor, scalar)
```
