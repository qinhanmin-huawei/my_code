# aten.scatter

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.scatter(
    Tensor self,
    int dim,
    Tensor index,
    Tensor src
) -> Tensor
```

### 1.2 算子功能
沿 `dim` 维按照 `index` 指定的位置，将 `src` 中的元素写入 `self` 的副本并返回新 Tensor，不修改输入 `self`。本规格使用 Tensor 类型的 `src`，由 overload packet 解析到 `aten.scatter.src`。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 提供初始值的输入 Tensor |
| `dim` | `int` | 执行 scatter 的维度，支持合法负维度 |
| `index` | `Tensor` | 写入位置索引，dtype 必须为 `int64` |
| `src` | `Tensor` | 待写入的数据 Tensor |
| 返回值 | `Tensor` | scatter 后的新 Tensor，shape 与 `self` 相同 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
本规格覆盖 1-4 维 Tensor。输出 shape 与 `self.shape` 相同；`index` 与 `src` 的维度数必须与 `self` 相同，且除 `dim` 外各维大小不能超过 `self` 对应维度。基础用例令 `index.shape == src.shape == self.shape`。

### 2.3 特殊限制说明
> `index.dtype` 必须为 `torch.int64`，其中的每个索引值必须位于 `[0, self.size(dim) - 1]`。`self` 与 `src` 的 dtype 和 device 应兼容。若同一输出位置出现重复索引且对应 `src` 值不同，结果可能具有非确定性；本规格用例使用唯一索引避免该情况。复数类型不纳入本规格考察。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (16, 32), (4, 8, 16), (2, 4, 8, 16)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_scatter(self, shape, dtype):
        def op_calc(tensor, dim, index, src):
            return torch.ops.aten.scatter(tensor, dim, index, src)

        def op_calc_1(tensor, dim, index, src):
            y = torch.ops.aten.scatter(tensor, dim, index, src)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        src = self._generate_tensor(shape, dtype)
        index_shape = (shape[0],) + (1,) * (len(shape) - 1)
        index = torch.arange(shape[0], dtype=torch.int64, device='npu')
        index = index.reshape(index_shape).expand(shape).clone()
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tensor, 0, index, src)
```