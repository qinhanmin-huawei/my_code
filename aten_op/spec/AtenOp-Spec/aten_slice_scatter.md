# aten.slice_scatter

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.slice_scatter(
    Tensor self,
    Tensor src,
    int dim=0,
    SymInt? start=None,
    SymInt? end=None,
    SymInt step=1
) -> Tensor
```

本文记录 Tensor 调用入口，对应 `aten.slice_scatter.default`。

### 1.2 算子功能

`aten.slice_scatter` 用 `src` 替换 `self` 沿 `dim` 的指定切片，其他位置保留原值，返回新张量。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 提供未更新区域的原始数据 |
| `src` | `Tensor` | 写入切片的数据，shape 必须与目标切片完全一致 |
| `dim` | `int` | 切片维度，默认 0，范围为 [-self.ndim, self.ndim - 1] |
| `start` | `SymInt?` | 起始位置，包含；None 表示从头开始 |
| `end` | `SymInt?` | 结束位置，不包含；None 表示到该维末尾 |
| `step` | `SymInt` | 正整数步长，默认 1 |
| 返回值 | `Tensor` | 更新后的新张量，与 self 同 shape、同 dtype |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self、src | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | self、src | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、`self` 的目标维度范围：[1-4]，待目标设备验证。

2、`src.shape` 必须等于目标切片 shape，不支持广播；输出 shape 与 `self` 相同。空切片要求 `src` 也具有对应的空 shape。

### 2.3 特殊限制说明

> `step > 0`，`start/end` 支持负索引和越界裁剪，`dim` 必须有效。当前规格要求 `self/src` 同 dtype、同设备，不修改输入张量。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    slice_cases = [
        (0, None, None, 1),
        (0, 1, -1, 1),
        (-1, 1, -1, 2),
        (0, -8, None, 2),
        (0, -10000, 10000, 1),
    ]
    shape_slice_cases = []
    for shape in shapes:
        for slice_args in slice_cases:
            shape_slice_cases.append((shape, slice_args))
        for dim in range(1, len(shape) - 1):
            shape_slice_cases.append((shape, (dim, 1, -1, 2)))

    @parametrize('shape,slice_args', shape_slice_cases)
    @parametrize('dtype', dtypes)
    def test_aten_slice_scatter(self, shape, dtype, slice_args):
        def op_calc(tensor, src, dim, start, end, step):
            y = torch.ops.aten.slice_scatter(tensor, src, dim, start, end, step)
            return y

        def op_calc_1(tensor, src, dim, start, end, step):
            y = torch.ops.aten.slice_scatter(tensor, src, dim, start, end, step)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        dim, start, end, step = slice_args
        src_shape = list(shape)
        src_shape[dim] = len(range(*slice(start, end, step).indices(shape[dim])))
        src = self._generate_tensor(tuple(src_shape), dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, src, dim, start, end, step)
```
