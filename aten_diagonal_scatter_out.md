# aten.diagonal_scatter.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.diagonal_scatter.out(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1, *, Tensor(a!) out) -> Tensor(a!)
```
### 1.2 算子功能
`aten.diagonal_scatter.out` 是 `aten.diagonal_scatter` 的 out 变体：将 `src` 张量的元素写入 `self` 中由 `offset`、`dim1`、`dim2` 确定的对角线位置，并将结果写入预分配的输出张量 `out`。该算子不会原地修改 `self`，`src` 的长度必须与目标对角线的长度一致。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 目标源张量，需要至少 2 维 |
| `src` | `Tensor` | 待写入对角线位置的源数据张量 |
| `offset` | `int` | 对角线偏移量，默认为 0（主对角线） |
| `dim1` | `int` | 参与写入的第一个维度，默认为 0 |
| `dim2` | `int` | 参与写入的第二个维度，默认为 1 |
| `out` | `Tensor(a!)` | 输出张量，需与 `self` 同形状同 dtype |
| 返回值 | `Tensor(a!)` | 即 `out` |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、shape维度范围：[2-4]（需至少 2 维，且 `dim1` 与 `dim2` 不同）


### 2.3 特殊限制说明

> `dim1`、`dim2` 需为有效且不同的维度。


### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(4, 8), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_diagonal_scatter_out(self, shape, dtype):
        def op_calc(tensor, tensor2):
            diag = torch.ops.aten.diagonal.default(tensor, 0, 1, 0)
            src = torch.ones_like(diag)
            out = torch.empty_like(tensor)
            torch.ops.aten.diagonal_scatter.out(tensor, src, 0, 1, 0, out=out)
            return out + out

        def op_calc_1(tensor, tensor2):
            diag = torch.ops.aten.diagonal.default(tensor, 0, 1, 0)
            src = torch.ones_like(diag)
            out = torch.empty_like(tensor)
            torch.ops.aten.diagonal_scatter.out(tensor, src, 0, 1, 0, out=out)
            return out

        tensor = self._generate_tensor(shape, dtype)
        tensor2 = torch.full(shape, 2, dtype=getattr(torch, dtype), device="npu")
        self.optype = 'VV'
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, tensor2)
```
