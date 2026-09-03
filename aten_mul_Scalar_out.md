# aten.mul.Scalar_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.mul.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能
`aten.mul.Scalar_out` 将输入张量 `self` 的每个元素与标量 `other` 相乘，并将结果写入调用者提供的 `out` 张量。输出形状与 `self` 相同；该操作会修改 `out`，不修改 `self`。

## 2. OP 规格

### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 参与乘法运算的源张量 |
| `other` | `Scalar`（`int` / `float` / `bool`） | 应用于输入张量每个元素的乘法标量 |
| `out` | `Tensor(a!)` | 保存结果的输出张量，形状须与 `self` 一致并会被修改 |
| 返回值 | `Tensor(a!)` | 返回 `out` 张量本身 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| ------ | ---- | ---- |  ----  |  ---- |  ----  | ----- | ----- | ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    | -      | -     | -      | √     | -     | √     | √     | √      | √      | √    |
| NPU    | -    | √    | -      | -     | -      | √     | -     | √     | √     | √      | √      | √    |

#### 2.2.2 Shape 支持
1、输入张量维度范围：[1-4]。

2、`out` 的形状须与 `self` 一致；标量不参与形状计算。
### 2.3 特殊限制说明
> `out` 的设备和数据类型应与输入及算子计算结果兼容；`uint*` 和 `int16` 类型不纳入支持范围。本文仅覆盖 `aten.mul.Scalar_out` 重载。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_mul_Scalar_out(self, shape, dtype):
        def op_calc(tensor, scalar, out_t):
            return torch.ops.aten.mul.Scalar_out(tensor, scalar, out=out_t)

        def op_calc_1(tensor, scalar, out_t):
            y = torch.ops.aten.mul.Scalar_out(tensor, scalar, out=out_t)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        scalar = self._generate_random_scalar(dtype)
        out_t = torch.empty_like(tensor)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tensor, scalar, out_t)
```