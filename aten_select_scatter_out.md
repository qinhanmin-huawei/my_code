# aten.select_scatter.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.select_scatter.out(Tensor self, Tensor src, int dim, SymInt index, *, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能
将 `self` 的内容复制到调用者提供的 `out`，再将 `src` 写入 `out` 沿 `dim` 维由 `index` 指定的单个切片，并返回 `out`。该重载修改 `out`，不修改 `self`。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入 Tensor |
| `src` | `Tensor` | 待写入切片，shape 等于 `self` 删除 `dim` 后的 shape |
| `dim` | `int` | 写入维度 |
| `index` | `SymInt` | 写入切片下标 |
| `out` | `Tensor(a!)` | 输出 Tensor，shape 与 `self` 相同，原地写入 |
| 返回值 | `Tensor(a!)` | 返回 `out` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
输入和 `out` 支持 1-4 维，`out.shape == self.shape`；`src.shape` 必须等于 `self.shape` 删除 `dim` 后的 shape。

### 2.3 特殊限制说明
> `out` 必须通过关键字参数传入，并与结果具有相同 shape、dtype 和 device。`dim`、`index` 和 `src.shape` 必须合法。该重载返回与 `out` 别名的 Tensor。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (16, 32), (4, 8, 16), (2, 4, 8, 16)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_select_scatter_out(self, shape, dtype):
        def op_calc(tensor, src, dim, index, out_tensor):
            y=torch.ops.aten.select_scatter.out(
                tensor, src, dim, index, out=out_tensor)
            return y+y

        def op_calc_1(tensor, src, dim, index, out_tensor):
            y = torch.ops.aten.select_scatter.out(
                tensor, src, dim, index, out=out_tensor)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        src = self._generate_tensor(shape[1:], dtype)
        out_tensor = torch.empty_like(tensor)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tensor, src, 0, 0, out_tensor)
```