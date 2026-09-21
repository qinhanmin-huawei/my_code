# aten.sym_stride.int

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sym_stride.int(Tensor self, int dim) -> SymInt
```

### 1.2 算子功能

返回指定轴相邻索引在存储中的步长：`y = self.stride(dim)`，单位为元素。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | strided 布局的输入张量 |
| `dim` | `int` | 查询轴，支持负索引 |
| 返回值 | `SymInt` | 指定轴的元素步长，不是字节数 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、`self` 的维度范围：[1-4]。

2、支持连续、步长为 2 的切片及广播视图；广播轴的 stride 可为 0。

3、支持二维连续输入在末轴长度动态变化时查询首轴步长。

### 2.3 特殊限制说明

> `dim` 范围为 `[-self.ndim, self.ndim - 1]`。静态步长查询可被常量化；步长依赖动态维度时，可保留为运行时参数。

### 2.4 使用方法

示例用 int64 Tensor 承载查询值，算子本身返回 `SymInt`；示例计时包含结果写入和后续运算，不代表查询算子的独立 kernel 性能。

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    cases = [dict(shape=shape, dim=0) for shape in shapes]
    shape_by_rank = {}
    for shape in shapes:
        shape_by_rank.setdefault(len(shape), tuple(shape))

    if 2 in shape_by_rank:
        shape = shape_by_rank[2]
        cases.extend([
            dict(shape=shape, dim=-1),
            dict(shape=shape, dim=-1, layout='strided'),
            dict(shape=shape, dim=0, layout='strided'),
            dict(shape=shape, dim=0, layout='expanded'),
            dict(shape=shape, dim=0, dynamic=True),
        ])
    if 3 in shape_by_rank:
        cases.append(dict(shape=shape_by_rank[3], dim=1))

    @parametrize('case', cases)
    @parametrize('dtype', dtypes)
    def test_aten_sym_stride_int(self, case, dtype):
        shape = case['shape']
        dim = case['dim']
        layout = case.get('layout', 'contiguous')
        dynamic = case.get('dynamic', False)

        def op_calc(tensor, dim, layout, shape):
            if layout == 'strided':
                tensor = tensor[..., ::2]
            elif layout == 'expanded':
                tensor = tensor.expand(shape)
            stride = torch.ops.aten.sym_stride.int(tensor, dim)
            y = torch.full((1,), stride, dtype=torch.int64, device=tensor.device)
            return y

        def op_calc_1(tensor, dim, layout, shape):
            if layout == 'strided':
                tensor = tensor[..., ::2]
            elif layout == 'expanded':
                tensor = tensor.expand(shape)
            stride = torch.ops.aten.sym_stride.int(tensor, dim)
            y = torch.full((1,), stride, dtype=torch.int64, device=tensor.device)
            return y + y

        shapes = [shape, (shape[0], shape[1] + 8)] if dynamic else [shape]
        for shape in shapes:
            source_shape = shape
            if layout == 'strided':
                source_shape = (*shape[:-1], shape[-1] * 2)
            elif layout == 'expanded':
                source_shape = (1, *shape[1:])
            tensor = self._generate_tensor(source_shape, dtype)
            if dynamic:
                torch._dynamo.mark_dynamic(tensor, 1)
            self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, dim, layout, shape)
```
