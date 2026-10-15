# aten.ne.Scalar_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能
将 Tensor 与标量逐元素比较，将 bool 结果写入 `out` Tensor。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 待比较 Tensor |
| `other` | `Scalar` | 对应元素比较标量 |
| `out` | `Tensor(a!)` | 写入 bool 比较结果 |
| 返回值 | `Tensor(a!)` | 返回 `out` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

`out` dtype 必须为 `bool`，shape 与广播后结果一致。

#### 2.2.2 Shape 支持
输入支持 1-4 维；标量不改变 shape。

### 2.3 特殊限制说明
> `out` 为原地写入参数，必须与输出设备、shape 和 bool dtype 兼容。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_ne_Scalar_out(self, shape, dtype):
        def op_calc(tensor, scalar, out_tensor):
            return torch.ops.aten.ne.Scalar_out(tensor, scalar, out=out_tensor)

        def op_calc_1(tensor, scalar, out_tensor):
            y = torch.ops.aten.ne.Scalar_out(tensor, scalar, out=out_tensor)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        scalar = self._generate_random_scalar(dtype)
        out_tensor = torch.empty(shape, dtype=torch.bool, device='npu')
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tensor, scalar, out_tensor)
```