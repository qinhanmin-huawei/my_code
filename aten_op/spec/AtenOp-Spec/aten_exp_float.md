# aten.exp.float

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.exp.float(float a) -> float
```
### 1.2 算子功能
`aten.exp.float` 是纯标量重载：对 float 标量 a 计算自然指数 e^a，返回 float。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `float` | 输入标量 |
| 返回值 | `float` | e^a 的结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | √ | √ | √ | - | - | - |
| NPU | - | - | - | √ | √ | √ | - | - | - |

> 纯标量重载：`a` 固定为 float，返回 float，与 tensor dtype/shape 无关，不参与 tensor 的 dtype 判定。

#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

> 纯标量重载在 `torch.compile(inductor)` 下被常量折叠。


### 2.4 使用方法


```python
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_exp_float(self, shape, dtype):
        def op_calc(first_element, scalar_a):
            y = torch.ops.aten.exp.float(scalar_a)
            result = first_element + y
            return result

        def op_calc_1(first_element, scalar_a):
            y = torch.ops.aten.exp.float(scalar_a)
            result = first_element + y + y
            return result

        first_element = torch.zeros(shape, dtype=getattr(torch, dtype), device="npu")
        scalar_a = self._generate_random_scalar(dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, scalar_a)
```
