# aten.new_empty.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.new_empty.out(
    Tensor self,
    SymInt[] size,
    *,
    Tensor(a!) out
) -> Tensor(a!)
```

### 1.2 算子功能
使 `out` 承接一个 shape 由 `size` 指定的未初始化 Tensor，并返回同一个 `out` Tensor。该重载不读取 `self` 的数据内容；输出存储和 dtype 由 `out` 确定。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 参考 Tensor，数据内容不参与计算 |
| `size` | `SymInt[]` | 输出 Tensor 的目标 shape |
| `out` | `Tensor(a!)` | 承接结果并被原地修改的输出 Tensor |
| 返回值 | `Tensor(a!)` | 返回与 `out` 具有相同别名的 Tensor |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
本规格覆盖 1-4 维 `self` 和 1-4 维输出 Tensor。`out` 的目标 shape 由 `size` 决定，不要求与 `self.shape` 相同；基础用例预先创建与 `size` shape 一致的 `out`。

### 2.3 特殊限制说明
> `out` 必须通过关键字参数传入。算子返回后 `out` 中的元素值未初始化，不能直接用于 eager 与 Inductor 数值比较；用例需先对返回 Tensor 进行确定性初始化。为避免自动 resize 引入额外行为，本规格用例要求 `out.shape == tuple(size)`。`out` 应位于目标设备并使用目标 dtype；复数类型不纳入本重载考察。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_new_empty_out(self, shape, dtype):
        def op_calc(tensor, size, out_tensor):
            result = torch.ops.aten.new_empty.out(
                tensor, size, out=out_tensor)
            return result.zero_()

        def op_calc_1(tensor, size, out_tensor):
            result = torch.ops.aten.new_empty.out(
                tensor, size, out=out_tensor)
            return result.zero_()

        tensor = self._generate_tensor(shape, dtype)
        size = tuple(shape)
        out_tensor = torch.empty(size, dtype=eval('torch.' + dtype),
                                 device='npu')
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tensor, size, out_tensor)
```