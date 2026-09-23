# aten.full.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.full.out(
    SymInt[] size,
    Scalar fill_value,
    *,
    Tensor(a!) out
) -> Tensor(a!)
```

### 1.2 算子功能
将 shape 由 `size` 指定、元素值为 `fill_value` 的 Tensor 写入调用者提供的 `out`，并返回 `out`。输出 dtype、layout 和 device 由 `out` 确定。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `size` | `SymInt[]` | 输出 Tensor 的目标 shape |
| `fill_value` | `Scalar` | 用于填充输出各元素的标量值 |
| `out` | `Tensor(a!)` | 接收结果并被原地修改的输出 Tensor |
| 返回值 | `Tensor(a!)` | 返回与 `out` 具有相同别名的 Tensor |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
本规格覆盖 1-4 维输出 Tensor，输出 shape 由 `size` 决定。基础用例预先创建与 `size` shape 一致的 `out`。

### 2.3 特殊限制说明
> `out` 必须通过关键字参数传入。`fill_value` 必须能转换为 `out.dtype`；超出其表示范围时可能报错或按 PyTorch 类型转换规则处理。为避免自动 resize 引入额外行为，基础用例要求 `out.shape == tuple(size)`，且 `out` 位于目标设备并使用目标 dtype。该重载会修改 `out`，返回值与 `out` 具有别名关系；NPU pinned memory 和复数类型不纳入本规格考察。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_full_out(self, shape, dtype):
        def op_calc(size, value, out_tensor):
            return torch.ops.aten.full.out(
                size, value, out=out_tensor)

        def op_calc_1(size, value, out_tensor):
            y = torch.ops.aten.full.out(
                size, value, out=out_tensor)
            return y + y

        value = True if dtype == 'bool' else 1
        out_tensor = torch.empty(
            shape, dtype=eval('torch.' + dtype), device='npu')
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tuple(shape), value, out_tensor)
```