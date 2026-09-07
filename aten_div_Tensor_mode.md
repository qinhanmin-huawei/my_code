# aten.div.Tensor_mode

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.div.Tensor_mode(self: Tensor, other: Tensor, *, rounding_mode: str) -> Tensor
```
### 1.2 算子功能
1、`aten.div.Tensor_mode` 是带舍入模式（rounding_mode）的逐元素除法算子，对输入张量 `self` 与 `other` 中对应元素执行除法，支持广播。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 被除数张量 |
| `other` | `Tensor` | 除数张量，须与 `self` 可广播（broadcastable） |
| `rounding_mode` | `str` | 舍入模式，取 `"trunc"`（截断）或 `"floor"`（向下取整）；为 None 时即真除法 |
| 返回值 | `Tensor` | 一个新的张量，形状为广播后的形状，dtype 与输入相同 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`self` 与 `other` 支持广播，输出张量形状为广播后的形状


### 2.3 特殊限制说明

> 1、`bool` 类型不支持：GPU eager 层报错（div_trunc/div_floor not implemented for Bool）。


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', _DTYPES_NO_BOOL)
    def test_aten_div_Tensor_mode(self, shape, dtype):
        def op_calc(first_element, second_element):
            y = torch.ops.aten.div.Tensor_mode(first_element, second_element, rounding_mode="trunc")
            return y

        def op_calc_1(first_element, second_element):
            y = torch.ops.aten.div.Tensor_mode(first_element, second_element, rounding_mode="trunc")
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype, min=1) if dtype in _INT_DTYPES else self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, second_element)
```
