# aten.eq.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.eq.default(Scalar a, Scalar b) -> bool
```
### 1.2 算子功能
1、`aten.eq.default` 是纯标量重载：对两个 Python 标量 `a`、`b` 做“等于”（`==`）比较，返回 Python `bool`。

2、与 tensor dtype / shape 无关，在 `torch.compile(inductor)` 下被常量折叠。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `Scalar` | 第一个比较数 |
| `b` | `Scalar` | 第二个比较数 |
| 返回值 | `bool` | 比较结果（Python `bool`） |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | √ | √ | √ | √ | √ | √ | - | - | - |
| NPU | √ | √ | √ | √ | √ | √ | - | - | - |

> 纯标量重载：参数为泛化 `Scalar`，返回 Python `bool`，与 tensor dtype / shape 无关。表中 dtype 为承载结果的载体 tensor 类型。

#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

1、 不接受 Python `bool` 实参：传 `True` / `False` 报 `INTERNAL ASSERT FAILED ... expected int`。

2、 纯标量重载在 inductor 下被常量折叠、不生成 kernel；用例靠同 dtype 的 zeros 载体承接结果。

3、 载体须写成 `载体 * 2 + int(y)`：`y` 取 False 时 `载体 + y` 会被 `remove_noop_ops` 整图消除，不产出 output_code。


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']

    @parametrize('dtype', dtypes)
    def test_aten_eq_default(self, dtype):
        def op_calc(first_element, scalar_a, scalar_b):
            y = torch.ops.aten.eq.default(scalar_a, scalar_b)
            result = first_element * 2 + int(y)
            return result

        def op_calc_1(first_element, scalar_a, scalar_b):
            y = torch.ops.aten.eq.default(scalar_a, scalar_b)
            result = first_element * 2 + int(y) + int(y)
            return result

        first_element = torch.zeros((128,), dtype=getattr(torch, dtype), device="npu")
        scalar_a = self._generate_random_scalar(dtype)
        scalar_b = self._generate_random_scalar(dtype)
        self.execute_inductor_test((128,), dtype, op_calc, op_calc_1, first_element, scalar_a, scalar_b)
```
