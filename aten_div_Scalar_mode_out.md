# aten.div.Scalar_mode_out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.div.Scalar_mode_out(self: Tensor, other: Scalar, *, str? rounding_mode, out: Tensor(a!)) -> Tensor(a!)
```

### 1.2 算子功能

将张量逐元素除以标量，并按 `rounding_mode` 对商取整。 结果写入 `out`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 被除数张量。 |
| `other` | `Scalar` | 非零除数。 |
| `rounding_mode` | `str \| None` | 支持 `"trunc"`、`"floor"` 或 `None`。 |
| `out` | `Tensor` | 承接结果的可写张量。 |
| `返回值` | `Tensor` | 与 `out` 别名相同的张量。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |


#### 2.2.2 Shape 支持

输入 Tensor 支持 1～4 维。


### 2.3 特殊限制说明

`out` 必须与输入位于同一设备并能接收结果；测试使用非零除数，并覆盖 `trunc` 与 `floor`。由于社区2.10版会将除法转化为乘倒数，而当除数无法除尽时，会变为近似值，导致高精度计算时会舍入错误，社区版在2.12版本修复了该问题，此处不做修改，与社区版保持一致。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'float32', 'bfloat16']
    shapes = [(32,), (32, 32), (32, 32, 32), (4, 8, 16, 32)]

    @parametrize('rounding_mode', ('trunc', 'floor'))
    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_div_scalar_mode_out(self, rounding_mode, shape, dtype):
        def op_calc(first_element, other, mode, out):
            x = first_element
            torch.ops.aten.div.Scalar_mode_out(x, other, rounding_mode=mode, out=out)
            return out

        def op_calc_1(first_element, other, mode, out):
            x = first_element
            torch.ops.aten.div.Scalar_mode_out(x, other, rounding_mode=mode, out=out)
            return out + out

        tensor1 = self._generate_tensor(shape, dtype)
        other = 3
        out = torch.empty_like(tensor1)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, other, rounding_mode, out
        )
```
