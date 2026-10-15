# aten.min.dim_min

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
```

### 1.2 算子功能
沿指定整数维度求最小值和索引，并将两者写入 `min` 和 `min_indices` 输出 Tensor。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入 Tensor |
| `dim` | `int` | 单个归约维度 |
| `keepdim` | `bool` | 是否保留归约维度 |
| `min` | `Tensor(a!)` | 写入最小值的输出 Tensor |
| `min_indices` | `Tensor(b!)` | 写入索引的输出 Tensor，dtype 为 `int64` |
| 返回值 | `(Tensor, Tensor)` | 返回值与两个 out Tensor 对应 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | × | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

`min_indices` 输出 dtype 必须为 `int64`。
aclnn不支持int8，仅与cpu做精度对比

#### 2.2.2 Shape 支持
`min` 和 `min_indices` shape 应与归约后的输出 shape 一致。
### 2.3 特殊限制说明
> `min` 和 `min_indices` 必须使用关键字参数传入，并且不应与 `self` 存在不兼容的别名。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_min_dim_min(self, shape, dtype):
        def op_calc(tensor, min_out, index_out):
            values, _ = torch.ops.aten.min.dim_min(
                tensor, 0, False, min=min_out, min_indices=index_out)
            return values

        def op_calc_1(tensor, min_out, index_out):
            values, _ = torch.ops.aten.min.dim_min(
                tensor, 0, False, min=min_out, min_indices=index_out)
            return values + values

        tensor = self._generate_tensor(shape, dtype)
        result_shape = shape[1:] if len(shape) > 1 else ()
        min_out = torch.empty(result_shape, dtype=eval('torch.' + dtype),
                              device='npu')
        index_out = torch.empty(result_shape, dtype=torch.int64, device='npu')
        self.execute_inductor_test(result_shape, dtype, op_calc, op_calc_1,
                                   tensor, min_out, index_out)
```