# aten.select_scatter

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.select_scatter(Tensor self, Tensor src, int dim, SymInt index) -> Tensor
```

### 1.2 算子功能
复制 `self`，并将 `src` 写入复制结果沿 `dim` 维、下标为 `index` 的单个切片，返回更新后的 Tensor。输入 `self` 不被修改。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 提供输出 shape 和未选中区域数据的输入 Tensor |
| `src` | `Tensor` | 写入选中切片的数据，shape 等于 `self` 删除 `dim` 后的 shape |
| `dim` | `int` | 写入维度，支持负维度 |
| `index` | `SymInt` | 写入切片下标 |
| 返回值 | `Tensor` | 写入后的新 Tensor，shape 与 `self` 相同 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
输入支持 1-4 维。`src.shape` 必须等于 `self.shape` 删除 `dim` 后的 shape，输出 shape 与 `self.shape` 相同。

### 2.3 特殊限制说明
> `dim` 和 `index` 必须合法，`src` 与 `self` 的 dtype、device 必须兼容。算子返回新 Tensor，不修改 `self`。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (16, 32), (4, 8, 16), (2, 4, 8, 16)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_select_scatter(self, shape, dtype):
        def op_calc(tensor, src, dim, index):
            y=torch.ops.aten.select_scatter(tensor, src, dim, index)
            return y + y

        def op_calc_1(tensor, src, dim, index):
            y = torch.ops.aten.select_scatter(tensor, src, dim, index)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        src_shape = shape[1:]
        src = self._generate_tensor(src_shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tensor, src, 0, 0)
```