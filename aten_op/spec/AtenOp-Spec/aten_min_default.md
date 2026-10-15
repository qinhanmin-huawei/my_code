# aten.min.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.min.default(Tensor self) -> Tensor
```

### 1.2 算子功能
返回输入 Tensor 全部元素中的最小值，结果为零维 Tensor。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入 Tensor |
| 返回值 | `Tensor` | 全量最小值，零维 Tensor |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | × | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
输入支持 1-4 维，结果为零维 Tensor。

### 2.3 特殊限制说明
> 不应对复数 Tensor 调用，且输入不应为空 Tensor。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_min_default(self, shape, dtype):
        def op_calc(tensor):
            return torch.ops.aten.min.default(tensor)

        def op_calc_1(tensor):
            y = torch.ops.aten.min.default(tensor)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test((), dtype, op_calc, op_calc_1, tensor)
```