# aten.squeeze_copy.out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.squeeze_copy.out(
    Tensor self, *, Tensor(a!) out
) -> Tensor(a!)
```

### 1.2 算子功能

删除输入中所有长度为 1 的维度，将结果写入 `out`，元素值不变。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `out` | `Tensor(a!)` | 独立输出张量，与压缩结果同 shape、同 dtype，与 self 同设备 |
| 返回值 | `Tensor(a!)` | 写入结果的 out，与 out 别名 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self、out | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | self、out | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、`self` 的维度范围：[1-4]，输出可为 0 维。

2、删除全部长度为 1 的维度；没有此类维度时 shape 不变。

### 2.3 特殊限制说明

> 不指定维度时，长度为 1 的 batch 维也会被删除。 `out` 与 `self` 的 dtype、设备一致，示例不覆盖输入输出存储重叠。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    shape_cases = list(shapes)

    # Derive extras from one framework shape per rank.
    shape_by_rank = {}
    for shape in shapes:
        shape_by_rank.setdefault(len(shape), tuple(shape))

    for rank, shape in shape_by_rank.items():
        if rank == 1:
            shape_cases.append((1,))
        elif rank == 2:
            shape_cases.append((1, shape[1]))
        elif rank == 4:
            shape_cases.append((shape[0], 1, shape[2], 1))

    @parametrize('shape', shape_cases)
    @parametrize('dtype', dtypes)
    def test_aten_squeeze_copy_out(self, shape, dtype):
        def op_calc(tensor, out):
            y = torch.ops.aten.squeeze_copy.out(tensor, out=out)
            return y

        def op_calc_1(tensor, out):
            y = torch.ops.aten.squeeze_copy.out(tensor, out=out)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        out_shape = tuple(size for size in shape if size != 1)
        out = torch.empty(out_shape, dtype=tensor.dtype, device=tensor.device)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, out)
```
