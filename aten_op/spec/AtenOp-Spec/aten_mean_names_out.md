# aten.mean.names_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.mean.names_out(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能
沿指定命名维度求平均值，并将结果写入 `out` Tensor。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 带维度名称的输入 Tensor |
| `dim` | `str[1]` | 待归约的命名维度名称 |
| `keepdim` | `bool` | 是否保留归约维度 |
| `dtype` | `ScalarType?` | 可选的计算和输出 dtype |
| `out` | `Tensor(a!)` | 保存结果的输出 Tensor |
| 返回值 | `Tensor(a!)` | 返回 `out` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | √ | √ |
| NPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | - | - |

#### 2.2.2 Shape 支持
输入支持 1-4 维，`out` shape 必须与命名维度归约后的结果一致。

### 2.3 特殊限制说明
> 输入必须携带 named tensor 维度。由于现有测试框架会对参数执行 `deepcopy`，而 named Tensor 暂不支持该操作，用例在测试方法中预先构造 named Tensor 和 `out`，并通过闭包传入算子；`op_calc` 不修改输入。当前 NPU TorchScript/eager 对 named Tensor 仍报 `NYI`，因此现有 Inductor 框架无法完成该重载的 lowering 验证。`out` 的设备、shape 和 dtype 应与归约结果兼容。

### 2.4 使用方法
```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_mean_names_out(self, shape, dtype):
        tensor = self._generate_tensor(shape, dtype)
        names = ('N',) + tuple(f'dim_{index}'
                               for index in range(1, len(shape)))
        named_input = tensor.refine_names(*names)
        result_shape = shape[1:]
        out_tensor = torch.empty(result_shape, dtype=eval('torch.' + dtype),
                                 device='npu')

        def op_calc():
            return torch.ops.aten.mean.names_out(named_input, ['N'], False,
                                                  out=out_tensor)

        def op_calc_1():
            y = torch.ops.aten.mean.names_out(named_input, ['N'], False,
                                               out=out_tensor)
            return y + y

        self.execute_inductor_test(result_shape, dtype, op_calc, op_calc_1)
```