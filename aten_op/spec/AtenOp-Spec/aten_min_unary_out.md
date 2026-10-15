# aten.min.unary_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.min.unary_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能
对输入 Tensor 的所有元素求最小值，并将零维结果写入 `out` Tensor。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入 Tensor |
| `out` | `Tensor(a!)` | 保存零维最小值的输出 Tensor |
| 返回值 | `Tensor(a!)` | 返回 `out` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | × | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
输入支持 1-4 维；`out` 必须为与结果兼容的零维 Tensor。

### 2.3 特殊限制说明
> `out` 的设备和 dtype 应与输入兼容，且必须为零维 Tensor；复数不纳入本重载考察。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_min_unary_out(self, shape, dtype):
        def op_calc(tensor, out_tensor):
            return torch.ops.aten.min.unary_out(tensor, out=out_tensor)

        def op_calc_1(tensor, out_tensor):
            y = torch.ops.aten.min.unary_out(tensor, out=out_tensor)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        out_tensor = torch.empty((), dtype=eval('torch.' + dtype), device='npu')
        self.execute_inductor_test((), dtype, op_calc, op_calc_1,
                                   tensor, out_tensor)
```