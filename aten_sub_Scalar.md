# aten.sub.Scalar

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.sub.Scalar(self: Tensor, other: Scalar, alpha: Scalar = 1) -> Tensor
```
### 1.2 算子功能
1、`aten.sub.Scalar` 是逐元素（Element-wise）的减法算子：对输入张量 `self` 中的每个元素减去标量 `other` 的对应倍数，即 out = self - alpha × other，默认 `alpha = 1`。
2、输出张量的形状与输入一致，dtype 与输入相同（同 dtype 输入时无类型提升）。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 被减数张量 |
| `other` | `Scalar` | 减数标量（Python 整数或浮点数） |
| `alpha` | `Scalar` | 减数的缩放系数，默认为 `1`，即 out = self - alpha × other |
| 返回值 | `Tensor` | 一个新的张量，形状与输入一致，dtype 与输入相同 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、输出张量与输入张量形状相同


### 2.3 特殊限制说明

> 1、`bool` 类型不支持（同 `sub.Tensor`）；整型输入时 `alpha` 不能为浮点数。
> 2、标量 `other` 的数值范围需与 tensor dtype 匹配（如 int8），避免溢出。


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_sub_Scalar(self, shape, dtype):
        def op_calc(first_element, scalar):
            y = torch.ops.aten.sub.Scalar(first_element, scalar)
            return y

        def op_calc_1(first_element, scalar):
            y = torch.ops.aten.sub.Scalar(first_element, scalar)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        scalar = 2
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, scalar)
```
