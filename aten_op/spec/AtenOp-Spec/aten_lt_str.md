# aten.lt.str

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.lt.str(str a, str b) -> bool
```
### 1.2 算子功能
1、`aten.lt.str` 是纯标量重载：对两个 Python `str` `a`、`b` 按字典序做“小于”（`<`）比较，返回 Python `bool`。

2、与 tensor dtype / shape 无关，在 `torch.compile(inductor)` 下被常量折叠。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `str` | 第一个字符串 |
| `b` | `str` | 第二个字符串 |
| 返回值 | `bool` | 比较结果（Python `bool`） |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | - | - | - | - |
| NPU | - | - | - | - | - | - | - |

> 该 overload 不涉及数值 dtype：参数与返回值均为 Python `str` / `bool`，表中无适用项；用例用固定的 float32 载体承接比较结果。

#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

1. 参数固定为 Python `str`，按字典序比较，与 tensor dtype 无关。
2. 纯标量重载在 `torch.compile(inductor)` 下被常量折叠，不生成 kernel；用例用固定的 float32 zeros 载体承接结果，由载体上的加法触发融合 kernel。

### 2.4 使用方法


```python
    def test_aten_lt_str(self):
        def op_calc(first_element, scalar_a, scalar_b):
            y = torch.ops.aten.lt.str(scalar_a, scalar_b)
            result = first_element + y
            return result

        def op_calc_1(first_element, scalar_a, scalar_b):
            y = torch.ops.aten.lt.str(scalar_a, scalar_b)
            result = first_element + y + y
            return result

        first_element = torch.zeros((128,), dtype=torch.float32, device="npu")
        scalar_a = "b"
        scalar_b = "a"
        self.execute_inductor_test((128,), "float32", op_calc, op_calc_1, first_element, scalar_a, scalar_b)
```
