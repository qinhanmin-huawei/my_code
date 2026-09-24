# aten.gather.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.gather.out(
    Tensor self,
    int dim,
    Tensor index,
    *,
    bool sparse_grad=False,
    Tensor(a!) out
) -> Tensor(a!)
```

### 1.2 算子功能
沿 `dim` 维使用 `index` 中的元素作为下标，从输入 `self` 收集数据并将结果写入 `out`，同时返回 `out`。输出 shape 应与 `index.shape` 相同；`self` 不被修改。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 提供被收集数据的输入 Tensor |
| `dim` | `int` | 收集所在维度，支持合法负维度 |
| `index` | `Tensor` | 收集下标，必须为 `torch.int64` |
| `sparse_grad` | `bool` | 是否使用稀疏梯度；默认为 `False` |
| `out` | `Tensor(a!)` | 接收收集结果并被修改的输出 Tensor |
| 返回值 | `Tensor(a!)` | 返回与 `out` 具有相同别名的 Tensor |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
输入 `self` 支持 1-4 维，`index.dim()` 必须等于 `self.dim()`。输出 shape 为 `index.shape`；除 `dim` 外，`index.size(d)` 不得大于 `self.size(d)`。基础用例预先创建与 `index.shape` 一致的 `out`。

### 2.3 特殊限制说明
> `index.dtype` 必须为 `torch.int64`，每个索引值必须位于 `[0, self.size(dim) - 1]`；`dim` 必须是合法维度。`out` 必须通过关键字参数传入，并与收集结果具有兼容的 dtype、device 和 shape。基础用例使用 `sparse_grad=False`，不考察反向传播和复数类型。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (16, 32), (4, 8, 16), (2, 4, 8, 16)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_gather_out(self, shape, dtype):
        def op_calc(tensor, dim, index, out_tensor):
            return torch.ops.aten.gather.out(
                tensor, dim, index, sparse_grad=False, out=out_tensor)

        def op_calc_1(tensor, dim, index, out_tensor):
            y = torch.ops.aten.gather.out(
                tensor, dim, index, sparse_grad=False, out=out_tensor)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        index_shape = (shape[0],) + (1,) * (len(shape) - 1)
        index = torch.arange(shape[0], dtype=torch.int64, device='npu')
        index = index.reshape(index_shape).expand(shape).clone()
        out_tensor = torch.empty(index.shape, dtype=eval('torch.' + dtype),
                                 device='npu')
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tensor, 0, index, out_tensor)
```