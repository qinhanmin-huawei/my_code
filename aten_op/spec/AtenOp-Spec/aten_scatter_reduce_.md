# aten.scatter_reduce_

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.scatter_reduce_(
    Tensor(a!) self,
    int dim,
    Tensor index,
    Tensor src,
    str reduce,
    *,
    bool include_self=True
) -> Tensor(a!)
```

### 1.2 算子功能
沿 `dim` 维按照 `index` 指定的位置，将 `src` 中的元素通过 `reduce` 指定的方式原地归约到 `self` 并返回 `self`。该 overload packet 对应 `aten.scatter_reduce_.two`。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor(a!)` | 被原地修改并提供可选归约初值的输入 Tensor |
| `dim` | `int` | 执行 scatter reduce 的维度，支持合法负维度 |
| `index` | `Tensor` | 归约位置索引，dtype 必须为 `int64` |
| `src` | `Tensor` | 待归约的数据 Tensor |
| `reduce` | `str` | 归约方式，可选 `sum`、`prod`、`mean`、`amax` 或 `amin` |
| `include_self` | `bool` | 是否将 `self` 原值包含在被写入位置的归约中，默认 `True` |
| 返回值 | `Tensor(a!)` | 返回被修改的 `self` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | × | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
本规格覆盖 1-4 维 Tensor。返回值 shape 与 `self.shape` 相同；`index` 与 `src` 的维度数必须与 `self` 相同。基础用例令三者 shape 一致。

### 2.3 特殊限制说明
> 该算子会修改 `self`，返回值与 `self` 具有别名关系。`index.dtype` 必须为 `torch.int64`，索引值必须在合法范围内。不同 `reduce` 模式对 dtype 的实际支持可能存在差异；本规格用例使用 `sum`。重复索引的浮点归约可能产生累加顺序误差，基础用例使用唯一索引。对需要梯度的叶子 Tensor 执行原地操作还会受到 autograd 规则限制。bool 和复数类型不纳入本规格考察。int8 在 GPU/NPU 的 native Triton 原子归约路径不支持（`atomic_add` 不支持 int8），如需 eager 语义通常只能依赖后端 fallback，故不列入本规格 native 支持范围。
### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32']
    shapes = [(128,), (16, 32), (4, 8, 16), (2, 4, 8, 16)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_scatter_reduce_(self, shape, dtype):
        def op_calc(tensor, dim, index, src):
            y=torch.ops.aten.scatter_reduce_(
                tensor, dim, index, src, 'sum', include_self=True)
            return y+y

        def op_calc_1(tensor, dim, index, src):
            y = torch.ops.aten.scatter_reduce_(
                tensor, dim, index, src, 'sum', include_self=True)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        src = self._generate_tensor(shape, dtype)
        index_shape = (shape[0],) + (1,) * (len(shape) - 1)
        index = torch.arange(shape[0], dtype=torch.int64, device='npu')
        index = index.reshape(index_shape).expand(shape).clone()
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tensor, 0, index, src)
```