# aten.le.float_int

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.le.float_int(float a, int b) -> bool
```
### 1.2 算子功能
1、`aten.le.float_int` 是纯标量重载：对 Python `float` `a` 与 `int` `b` 做“小于等于”（`<=`）比较，返回 Python `bool`。

2、与 tensor dtype / shape 无关，在 `torch.compile(inductor)` 下被常量折叠。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `float` | 第一个比较数 |
| `b` | `int` | 第二个比较数 |
| 返回值 | `bool` | 比较结果（Python `bool`） |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | √ | √ | √ | - |
| NPU | - | - | - | √ | √ | √ | - |

> 纯标量重载：参数为 Python 标量，返回 Python `bool`，与 tensor dtype / shape 无关。表中 dtype 为承载结果的载体 tensor 类型。

#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

1. 实参类型与顺序不能对调：传 `(3, 2.5)` 会报 `Expected a value of type 'float' for argument 'a' but instead found type 'int'`。
2. 纯标量重载在 `torch.compile(inductor)` 下被常量折叠，不生成 kernel；用例用同 dtype 的 zeros 载体承接结果，由载体上的加法触发融合 kernel。

### 2.4 使用方法


```python
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('dtype', dtypes)
    def test_aten_le_float_int(self, dtype):
        def op_calc(first_element, scalar_a, scalar_b):
            y = torch.ops.aten.le.float_int(scalar_a, scalar_b)
            result = first_element + y
            return result

        def op_calc_1(first_element, scalar_a, scalar_b):
            y = torch.ops.aten.le.float_int(scalar_a, scalar_b)
            result = first_element + y + y
            return result

        first_element = torch.zeros((128,), dtype=getattr(torch, dtype), device="npu")
        scalar_a = float(self._generate_random_scalar(dtype))  # schema: float
        scalar_b = int(self._generate_random_scalar(dtype))    # schema: int
        self.execute_inductor_test((128,), dtype, op_calc, op_calc_1, first_element, scalar_a, scalar_b)
```
