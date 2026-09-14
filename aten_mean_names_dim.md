# aten.mean.names_dim

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.mean.names_dim(Tensor self, Dimname[1]? dim, bool keepdim=False) -> Tensor
```

### 1.2 算子功能
沿指定命名维度求平均值；`keepdim=True` 时保留命名维度为 1。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 带维度名称的输入 Tensor |
| `dim` | `Dimname[]?` | 待归约的命名维度 |
| `keepdim` | `bool` | 是否保留归约维度 |
| 返回值 | `Tensor` | 按命名维度归约后的结果 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | √ | √ |
| NPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | - | - |

#### 2.2.2 Shape 支持
输入支持 1-4 维；输出 shape 按命名维度归约规则确定。

### 2.3 特殊限制说明
> 输入必须携带与 `dim` 匹配的 named tensor 维度。由于现有测试框架会对参数执行 `deepcopy`，而 named Tensor 暂不支持该操作，用例在测试方法中预先构造 named Tensor，并通过闭包传入算子；`op_calc` 不修改输入。NPU 复数类型不纳入本重载考察。

### 2.4 使用方法
```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_mean_names_dim(self, shape, dtype):
        tensor = self._generate_tensor(shape, dtype)
        names = ('N',) + tuple(f'dim_{index}'
                               for index in range(1, len(shape)))
        named_input = tensor.refine_names(*names)

        def op_calc():
            return torch.ops.aten.mean.names_dim(named_input, ['N'], False)

        def op_calc_1():
            y = torch.ops.aten.mean.names_dim(named_input, ['N'], False)
            return y + y

        result_shape = shape[1:]
        self.execute_inductor_test(result_shape, dtype, op_calc, op_calc_1)
```