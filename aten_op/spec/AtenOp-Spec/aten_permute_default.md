# aten.permute.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.permute.default(Tensor(a) self, int[] dims) -> Tensor(a)
```

### 1.2 算子功能
按照 `dims` 指定的维度顺序返回 `self` 的视图。该算子只重排 Tensor 的尺寸和 stride，不复制数据；返回 Tensor 与输入 `self` 共享底层存储。

例如：
```python
x.shape == (2, 3, 4)
torch.ops.aten.permute.default(x, [2, 0, 1]).shape == (4, 2, 3)
```

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor(a)` | 待重排维度的输入 Tensor |
| `dims` | `int[]` | 输出维度顺序；必须是 `0..self.dim()-1` 的一个排列 |
| 返回值 | `Tensor(a)` | 按 `dims` 重排后的 Tensor 视图，与 `self` 共享存储 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

复数类型不纳入 NPU 本规格考察。

#### 2.2.2 Shape 支持
本规格覆盖 1-4 维输入 Tensor。`dims` 长度必须等于 `self.dim()`；若输入 shape 为 `(s0, s1, ..., sn)`，则输出 shape 为：
```text
(s[dims[0]], s[dims[1]], ..., s[dims[n-1]])
```

输出 Tensor 的元素数量与输入相同，但 stride 按 `dims` 重排，通常为非连续视图。

### 2.3 特殊限制说明
> `dims` 必须是不重复且覆盖全部输入维度的整数排列，不能遗漏或重复维度；负维度不作为本规格用例。`permute` 返回 view，不保证 contiguous，若下游算子要求连续存储，应显式调用 `contiguous()`。由于该算子本身通常只产生视图元数据，不一定单独生成设备 kernel，本用例通过 `permute` 后的逐元素加法验证 Inductor 对该视图的处理。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (16, 32), (4, 8, 16), (2, 4, 8, 16)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_permute_default(self, shape, dtype):
        dims = tuple(reversed(range(len(shape))))
        result_shape = tuple(shape[index] for index in dims)

        def op_calc(tensor, perm_dims):
            return torch.ops.aten.permute.default(tensor, perm_dims)

        def op_calc_1(tensor, perm_dims):
            y = torch.ops.aten.permute.default(tensor, perm_dims)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(result_shape, dtype, op_calc, op_calc_1,
                                   tensor, dims)
```