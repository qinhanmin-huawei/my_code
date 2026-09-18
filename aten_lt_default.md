# aten.lt.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.lt.default(Scalar a, Scalar b) -> bool
```
### 1.2 算子功能
1、`aten.lt.default` 是纯标量重载：对两个 Python 标量 `a`、`b` 做“小于”（`<`）比较，返回 Python `bool`。

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

|        | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | √ | √ | √ | √ | √ | √ | - |
| NPU | √ | √ | √ | √ | √ | √ | - |

> 纯标量重载：参数为 Python 标量，返回 Python `bool`，与 tensor dtype / shape 无关。表中 dtype 为承载结果的载体 tensor 类型。

#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

1. 参数为泛化 `Scalar`：传 Python `bool` 会在 PyTorch 侧触发 `INTERNAL ASSERT FAILED ... expected int` 崩溃，故用例排除 bool 载体。
2. 纯标量重载在 `torch.compile(inductor)` 下被常量折叠，不生成 kernel；用例用同 dtype 的 zeros 载体承接结果，由载体上的加法触发融合 kernel。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']

    @parametrize('dtype', dtypes)
    def test_aten_lt_default(self, dtype):
        def op_calc(first_element, scalar_a, scalar_b):
            y = torch.ops.aten.lt.default(scalar_a, scalar_b)
            result = first_element + y
            return result

        def op_calc_1(first_element, scalar_a, scalar_b):
            y = torch.ops.aten.lt.default(scalar_a, scalar_b)
            result = first_element + y + y
            return result

        first_element = torch.zeros((128,), dtype=getattr(torch, dtype), device="npu")
        scalar_a = self._generate_random_scalar(dtype)
        scalar_b = self._generate_random_scalar(dtype)
        self.execute_inductor_test((128,), dtype, op_calc, op_calc_1, first_element, scalar_a, scalar_b)
```
