# aten._int_mm.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten._int_mm.default(self: Tensor, mat2: Tensor) -> Tensor
```
### 1.2 算子功能
`aten._int_mm.default` 是 `aten._int_mm` 的默认重载（default overload），其功能与 `aten._int_mm` 一致：执行纯整数（int8）矩阵乘法，计算 `self` 与 `mat2` 两个二维 int8 矩阵的乘积，返回一个 int32 类型的输出矩阵，常用于量化场景中矩阵乘的整数内核加速。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 左乘矩阵，二维 int8 张量，形状为 `(M, K)` |
| `mat2` | `Tensor` | 右乘矩阵，二维 int8 张量，形状为 `(K, N)` |
| 返回值 | `Tensor` | 乘积结果，int32 张量，形状为 `(M, N)` |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | - | - | - | - | - | - | × |
| NPU | - | √ | - | - | - | - | - | - | - | - | - | × |

#### 2.2.2 Shape 支持

1、输入为二维矩阵，`self` 形状 `(M, K)`，`mat2` 形状 `(K, N)`

2、`self.shape[1]` 必须等于 `mat2.shape[0]`

### 2.3 特殊限制说明

> 两个输入张量均必须为 int8 类型，输出为 int32 类型。输入必须为二维矩阵，且内维度必须匹配。

### 2.4 使用方法

```python
    dtypes = ['int8']
    shapes = [(128, 128)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__int_mm_default(self, shape, dtype):
        def op_calc(input, mat2):
            y = torch.ops.aten._int_mm.default(input, mat2)
            return y
        def op_calc_1(input, mat2):
            y = torch.ops.aten._int_mm.default(input, mat2)
            return y + y
        input = self._generate_tensor(shape, dtype)
        mat2 = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input, mat2)
```
