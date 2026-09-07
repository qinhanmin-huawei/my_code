# aten.mul.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能
`aten.mul.out` 对两个输入 Tensor 执行逐元素乘法，并将结果写入指定的 `out` Tensor。输入支持广播；`out` 会被修改并作为返回值返回。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 第一个输入张量 |
| `other` | `Tensor` | 第二个输入张量，须与 `self` 形状相同或可广播 |
| `out` | `Tensor(a!)` | 保存结果的输出张量，会被修改 |
| 返回值 | `Tensor(a!)` | 返回 `out` 张量本身 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| ------ | ---- | ---- |  ----  |  ---- |  ----  | ----- | ----- | ----- | ----- | ------ | ------ | ------ | --------- | ---------- |
| GPU    | -    | √    | -      | -     | -      | √     | -     | √     | √     | √      | √      | √      | -         | -          |
| NPU    | -    | √    | -      | -     | -      | √     | -     | √     | √     | √      | √      | √      | -         | -          |

#### 2.2.2 Shape 支持
1、输入 Tensor 维度范围：[1-4]。

2、`self` 与 `other` 须形状相同或满足广播规则；`out` 的形状须与广播结果一致。

3、输入参数 `self`、`other` 以及输出参数 `out` 支持 `int8`、`int32`、`int64`、`float16`、`float32`、`bfloat16`、`bool`、`complex64`；`complex128` 不支持。
### 2.3 特殊限制说明
> `out` 的设备和数据类型应与计算结果兼容；`complex128` 不支持。本文仅覆盖 `aten.mul.out` 重载。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64',
              'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_mul_out(self, shape, dtype):
        def op_calc(first_tensor, second_tensor, out_tensor):
            return torch.ops.aten.mul.out(first_tensor, second_tensor,
                                           out=out_tensor)

        def op_calc_1(first_tensor, second_tensor, out_tensor):
            y = torch.ops.aten.mul.out(first_tensor, second_tensor,
                                       out=out_tensor)
            return y + y

        first_tensor = self._generate_tensor(shape, dtype)
        second_tensor = self._generate_tensor(shape, dtype)
        out_tensor = torch.empty_like(first_tensor)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   first_tensor, second_tensor, out_tensor)
```