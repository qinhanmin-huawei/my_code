# aten.scalar_tensor.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.scalar_tensor.out(
    Scalar s,
    *,
    Tensor(a!) out
) -> Tensor(a!)
```

### 1.2 算子功能
将 Python 标量 `s` 转换为零维 Tensor，并将结果写入调用者提供的 `out` Tensor。该操作修改并返回 `out`，输出的 dtype、device 和 layout 由 `out` 决定。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `s` | `Scalar` | 待写入 `out` 的 Python 数值或布尔标量 |
| `out` | `Tensor(a!)` | 承接结果并被原地修改的零维输出 Tensor |
| 返回值 | `Tensor(a!)` | 返回与 `out` 具有相同别名的 Tensor |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |


#### 2.2.2 Shape 支持
结果固定为零维 Tensor，基础用例要求 `out.shape == ()`。

### 2.3 特殊限制说明
> `out` 必须通过关键字参数传入，且 `s` 必须能转换为 `out.dtype`。`out` 应位于目标设备并预先创建为零维 Tensor；若传入其他 shape，运行时可能尝试 resize，并产生额外行为或警告。该重载修改 `out`，返回值与 `out` 具有别名关系。复数类型不纳入本规格考察。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']

    @parametrize('dtype', dtypes)
    def test_aten_scalar_tensor_out(self, dtype):
        def op_calc(value, out_tensor):
            return torch.ops.aten.scalar_tensor.out(
                value, out=out_tensor)

        def op_calc_1(value, out_tensor):
            y = torch.ops.aten.scalar_tensor.out(
                value, out=out_tensor)
            return y + y

        value = True if dtype == 'bool' else 1
        out_tensor = torch.empty((), dtype=eval('torch.' + dtype),
                                 device='npu')
        self.execute_inductor_test((), dtype, op_calc, op_calc_1,
                                   value, out_tensor)
```