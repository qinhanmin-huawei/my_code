# aten.bitwise_and_.Scalar

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.bitwise_and_.Scalar(self: Tensor(a!), other: Scalar) -> Tensor(a!)
```
### 1.2 算子功能
1、`aten.bitwise_and_.Scalar` 是 `aten.bitwise_and` 在 `other` 为标量时的原地版本：逐元素按位与，结果直接写回 `self`。

2、不分配新张量，返回修改后的 `self` 本身。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor(a!)` | 源张量，须为整型或 bool，**会被原地修改** |
| `other` | `Scalar` | 参与按位与的标量，须与 `self` 的 dtype 匹配（bool 档传 `True`） |
| 返回值 | `Tensor(a!)` | 修改后的 `self` 本身 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | √      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、原地写回 `self`，形状不变。

### 2.3 特殊限制说明

1、 仅支持整数与 bool：float/bf16/fp16 输入在 eager 报 `NotImplementedError: "bitwise_and_cpu" not implemented for 'Half'/'BFloat16'/'Float'`。

2、 标量实参须与 `self` 的 dtype 匹配：bool 档须传 Python `True`。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_bitwise_and__Scalar(self, shape, dtype):
        def op_calc(first_element, scalar):
            y = torch.ops.aten.bitwise_and_.Scalar(first_element, scalar)
            return y

        def op_calc_1(first_element, scalar):
            y = torch.ops.aten.bitwise_and_.Scalar(first_element, scalar)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        # bool 档须传 Python True：否则结果提升为 int64，原地写回直接失败
        scalar = True if dtype == 'bool' else 5
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, scalar)
```
