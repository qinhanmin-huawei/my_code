# aten.min.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.min.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能
对 `self` 和 `other` 执行逐元素最小值运算，将结果写入 `out`。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 第一个输入 Tensor |
| `other` | `Tensor` | 第二个输入 Tensor，可按广播规则参与计算 |
| `out` | `Tensor(a!)` | 保存逐元素最小值的输出 Tensor |
| 返回值 | `Tensor(a!)` | 返回 `out` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | × | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
支持 1-4 维和 PyTorch 广播规则；`out` shape 应与广播后的结果一致。

### 2.3 特殊限制说明
> `out` 的设备、dtype 和 shape 应与结果兼容；复数不纳入本重载考察。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_min_out(self, shape, dtype):
        def op_calc(first, second, out_tensor):
            return torch.ops.aten.min.out(first, second, out=out_tensor)

        def op_calc_1(first, second, out_tensor):
            y = torch.ops.aten.min.out(first, second, out=out_tensor)
            return y + y

        first = self._generate_tensor(shape, dtype)
        second = self._generate_tensor(shape, dtype)
        out_tensor = torch.empty_like(first)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   first, second, out_tensor)
```