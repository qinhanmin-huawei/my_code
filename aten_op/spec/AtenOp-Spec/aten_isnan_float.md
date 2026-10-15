# aten.isnan.float

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.isnan.float(float a) -> bool
```
### 1.2 算子功能
1、`aten.isnan.float` 是纯标量重载：判断 Python `float` `a` 是否为 NaN，返回 Python `bool`。

2、与 tensor dtype / shape 无关，在 `torch.compile(inductor)` 下被常量折叠。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `float` | 待判断的标量 |
| 返回值 | `bool` | 判断结果（Python `bool`） |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | √ | √ | √ | - | - | - |
| NPU | - | - | - | √ | √ | √ | - | - | - |

> 纯标量重载：参数固定为 Python `float`，返回 Python `bool`，与 tensor dtype / shape 无关。表中 dtype 为承载结果的载体 tensor 类型。

#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

1、 有限值恒为 False，只有 `nan` 才是 True。

2、 常量折叠后不生成 kernel，用例用同 dtype 的 zeros 载体承接结果。

### 2.4 使用方法


```python
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('dtype', dtypes)
    def test_aten_isnan_float(self, dtype):
        def op_calc(first_element, scalar_a):
            y = torch.ops.aten.isnan.float(scalar_a)
            return first_element + y

        def op_calc_1(first_element, scalar_a):
            y = torch.ops.aten.isnan.float(scalar_a)          # 有限值 -> False
            z = torch.ops.aten.isnan.float(float('nan'))      # NaN -> True
            return first_element + y + z

        first_element = torch.zeros((128,), dtype=getattr(torch, dtype), device="npu")
        scalar_a = self._generate_random_scalar(dtype)
        self.execute_inductor_test((128,), dtype, op_calc, op_calc_1, first_element, scalar_a)
```
