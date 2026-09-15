# aten.min.dim

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
```

### 1.2 算子功能
沿指定的单个整数维度求最小值，同时返回最小值 `values` 和其在归约维度内的索引 `indices`。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入 Tensor |
| `dim` | `int` | 单个归约维度 |
| `keepdim` | `bool` | 是否保留该维度，默认 `False` |
| 返回值 | `(Tensor, Tensor)` | `values` 与 `indices`；索引 dtype 为 `int64` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | × | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

`indices` 输出 dtype 固定为 `int64`。
aclnn不支持int8，仅与cpu做精度对比
#### 2.2.2 Shape 支持
`values` 和 `indices` 具有相同 shape；`keepdim=False` 删除 `dim`，`keepdim=True` 将该维度设为 1。

### 2.3 特殊限制说明
> 返回值是二元组，应分别取用 `values, indices = ...`。现有 Inductor 测试中可仅校验 `values`，索引应另行验证。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_min_dim(self, shape, dtype):
        def op_calc(tensor):
            values, _ = torch.ops.aten.min.dim(tensor, 0, False)
            return values

        def op_calc_1(tensor):
            values, _ = torch.ops.aten.min.dim(tensor, 0, False)
            return values + values

        tensor = self._generate_tensor(shape, dtype)
        result_shape = shape[1:] if len(shape) > 1 else ()
        self.execute_inductor_test(result_shape, dtype, op_calc,
                                   op_calc_1, tensor)
```