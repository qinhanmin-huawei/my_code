# aten.bitwise_xor.Scalar

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.bitwise_xor.Scalar(self: Tensor, other: Scalar) -> Tensor
```
### 1.2 算子功能
1、`aten.bitwise_xor.Scalar` 是 `other` 为标量时的版本：对 `self` 每个元素与标量 `other` 按位异或。

2、结果 dtype 由标量实参的类型决定：bool 档须传 Python `True`。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 源张量，须为整型或 bool |
| `other` | `Scalar` | 参与按位异或的标量，须与 `self` 的 dtype 匹配（bool 档传 `True`） |
| 返回值 | `Tensor` | 新张量，dtype 与输入一致，形状与输入相同 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | √      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、输出形状与输入相同。

### 2.3 特殊限制说明

1、 仅支持整数与 bool：float/bf16/fp16 输入在 eager 报 `NotImplementedError: "bitwise_xor_cpu" not implemented for 'Half'/'BFloat16'/'Float'`。

2、 标量实参须与 `self` 的 dtype 匹配：bool 档须传 Python `True`。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_bitwise_xor_Scalar(self, shape, dtype):
        def op_calc(first_element, scalar):
            y = torch.ops.aten.bitwise_xor.Scalar(first_element, scalar)
            return y

        def op_calc_1(first_element, scalar):
            y = torch.ops.aten.bitwise_xor.Scalar(first_element, scalar)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        # bool 档须传 Python True：传 int 会把结果提升成 int64
        scalar = True if dtype == 'bool' else 5
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, scalar)
```
