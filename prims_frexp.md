# prims.frexp

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.prims.frexp(self: Tensor) -> (Tensor mantissa, Tensor exponent)
```
### 1.2 算子功能
`prims.frexp` 将输入张量 `self` 中的每个浮点数分解为**尾数（mantissa）** 和**指数（exponent）** 两部分，满足：
$$\text{self} = \text{mantissa} \times 2^{\text{exponent}}$$
- 尾数（mantissa）的绝对值在 `[0.5, 1)` 范围内（对于 `0` 则尾数为 `0`）。
- 指数（exponent）为整数（`int32` 类型）。
- 特殊值处理：
  - `0`：尾数和指数均为 `0`。
  - `inf` / `-inf`：尾数为自身，指数为 `0`。
  - `nan`：尾数为 `nan`，指数为 `0`。
- 返回一个元组 `(mantissa, exponent)`，两个张量的形状与输入张量相同。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 输入张量，须为浮点类型（`fp16`、`fp32`、`fp64`、`bf16`） |
| 返回值 | `Tuple[Tensor, Tensor]` | 返回 `(mantissa, exponent)` 元组，其中 `mantissa` 的 dtype 与输入相同，`exponent` 的 dtype 为 `torch.int32` |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | ×    |  -     |  -    |  -    | ×     | -      | ×      | √     | √      | √      | ×      |
| NPU    | -    | ×    |  -     |  -    |  -    | ×     | -      | ×      | √     | √      | √      | ×      |


#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_prims_frexp(self, shape, dtype):
        def op_calc_prims_frexp( first_element):
            mantissa, exponent = torch.ops.prims.frexp(first_element)
            mantissa =mantissa
            exponent =exponent
            return mantissa, exponent
        def op_calc_prims_frexp_1(first_element):
            mantissa, exponent = torch.ops.prims.frexp(first_element)
            mantissa =mantissa +1
            exponent =exponent +1
            return mantissa, exponent
        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc_prims_frexp, op_calc_prims_frexp_1, tensor)
```
