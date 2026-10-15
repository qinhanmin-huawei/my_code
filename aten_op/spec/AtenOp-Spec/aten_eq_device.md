# aten.eq.device

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.eq.device(Device a, Device b) -> bool
```
### 1.2 算子功能
1、`aten.eq.device` 是纯标量重载：比较两个 `torch.device` 是否相同，返回 Python `bool`。

2、与 tensor dtype / shape 无关，在 `torch.compile(inductor)` 下被常量折叠。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `Device` | 第一个设备 |
| `b` | `Device` | 第二个设备 |
| 返回值 | `bool` | 比较结果（Python `bool`） |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | √ | √ | √ | √ | √ | √ | - | - | - |
| NPU | √ | √ | √ | √ | √ | √ | - | - | - |

> 纯标量重载：参数为设备，返回 Python `bool`，与 tensor dtype / shape 无关。表中 dtype 为承载结果的载体 tensor 类型。

#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

1、 `Device` 即 `torch.device`，裸字符串 `"cpu"` 也被接受。

2、 bool 列为 `-` 是载体所致：`zeros(bool) * 2` 提升为 int64。

3、 纯标量重载被常量折叠、不生成 kernel；载体须写成 `载体 * 2 + int(y)`。


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']

    @parametrize('dtype', dtypes)
    def test_aten_eq_device(self, dtype):
        def op_calc(first_element):
            y = torch.ops.aten.eq.device(first_element.device, torch.device("cpu"))
            result = first_element * 2 + int(y)
            return result

        def op_calc_1(first_element):
            y = torch.ops.aten.eq.device(first_element.device, torch.device("cpu"))
            result = first_element * 2 + int(y) + int(y)
            return result

        first_element = torch.zeros((128,), dtype=getattr(torch, dtype), device="npu")
        self.execute_inductor_test((128,), dtype, op_calc, op_calc_1, first_element)
```
