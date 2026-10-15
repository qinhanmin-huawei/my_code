# aten.diagonal_copy.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.diagonal_copy.out(Tensor self, int offset=0, int dim1=0, int dim2=1, *, Tensor(a!) out) -> Tensor(a!)
```
### 1.2 算子功能
`aten.diagonal_copy.out` 是 `aten.diagonal_copy` 的 out 变体，提取输入张量 `self` 中由 `dim1`、`dim2` 两个维度与 `offset` 共同确定的对角线元素，并将结果写入预分配的输出张量 `out`。返回结果与原张量相互独立。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 源张量，需要至少 2 维 |
| `offset` | `int` | 对角线偏移量，默认为 0（主对角线）；正值取主对角线右上方，负值取左下方 |
| `dim1` | `int` | 参与取对角的第一个维度，默认为 0 |
| `dim2` | `int` | 参与取对角的第二个维度，默认为 1 |
| `out` | `Tensor(a!)` | 输出张量 |
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

> `dim1` 和 `dim2` 必须指向不同的有效维度，且都在 `[-self.dim(), self.dim())` 范围内。
>
> NPU 环境下建议显式传入全部参数（不依赖默认值），避免参数解析顺序差异导致 `dim1`/`dim2` 取值错误。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(4, 8), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_diagonal_copy_out(self, shape, dtype):
        def op_calc(tensor, tensor2):
            y = torch.ops.aten.diagonal_copy.default(tensor, 0, 1, 0)
            out = torch.empty_like(y)
            torch.ops.aten.diagonal_copy.out(tensor, 0, 1, 0, out=out)
            return out + out

        def op_calc_1(tensor, tensor2):
            y = torch.ops.aten.diagonal_copy.default(tensor, 0, 1, 0)
            out = torch.empty_like(y)
            torch.ops.aten.diagonal_copy.out(tensor, 0, 1, 0, out=out)
            return out

        tensor = self._generate_tensor(shape, dtype)
        tensor2 = torch.full(shape, 2, dtype=getattr(torch, dtype), device="npu")
        self.optype = 'VV'
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, tensor2)
```
