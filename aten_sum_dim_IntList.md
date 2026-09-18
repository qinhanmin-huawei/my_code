# aten.sum.dim_IntList

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sum.dim_IntList(
    Tensor self,
    int[1]? dim,
    bool keepdim=False,
    *,
    ScalarType? dtype=None
) -> Tensor
```

### 1.2 算子功能

沿指定维度求和：`y = sum(self, dim, keepdim)`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `dim` | `int[1]?` | 归约维度；支持负轴和多轴，None 或空列表表示全部维度 |
| `keepdim` | `bool` | 是否保留被归约维度，默认 False |
| `dtype` | `ScalarType?` | 可选的归约结果类型；指定时输入按该类型参与求和 |
| 返回值 | `Tensor` | 沿指定维度求和后的张量 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

未指定 `dtype` 时，整数和 bool 输入返回 int64，浮点输入保持原 dtype；显式 fp32 作为补充规格。

#### 2.2.2 Shape 支持

1、`self` 支持的维度范围：[1-4]。

2、`keepdim=False` 时删除被归约维度，`True` 时将其长度置为 1；全部维度被删除时输出为零维张量。

3、支持非整齐尺寸、步长为 2 的非连续输入、空归约及长度为 1 的归约轴；空归约的结果为零。

### 2.3 特殊限制说明

> `dim` 必须有效且不能重复。浮点求和可能受累加顺序及输出类型舍入影响，`dtype` 不代表后端内部累加精度。

### 2.4 使用方法

示例需先执行 `from itertools import product`；`_make_sum_input` 统一准备普通、步长布局及大整数用例的输入。

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    shape_by_rank = {}
    for shape in shapes:
        shape_by_rank.setdefault(len(shape), tuple(shape))

    dim_configs = [dict(shape=shape, dim=[-1]) for shape in shapes]
    if 2 in shape_by_rank:
        shape_2d = shape_by_rank[2]
        dim_configs.extend([
            dict(shape=shape_2d, dim=None),
            dict(shape=shape_2d, dim=[], keepdim=True),
            dict(shape=shape_2d, dim=[0], keepdim=True),
            dict(shape=shape_2d, dim=[1, 0]),
            dict(shape=shape_2d, dim=[-1], keepdim=True, result_dtype=torch.float32),
        ])

        irregular_shape = (shape_2d[0] + 1, max(1, shape_2d[1] - 1))
        empty_shape = (shape_2d[0], 0)
        singleton_shape = (shape_2d[0], 1)
        dim_configs.extend([
            dict(shape=irregular_shape, dim=[-1]),
            dict(shape=empty_shape, dim=[-1]),
            dict(shape=singleton_shape, dim=[-1]),
        ])

        dim_configs.append(dict(shape=shape_2d, dim=[-1], strided=True))

    if 3 in shape_by_rank:
        shape_3d = shape_by_rank[3]
        dim_configs.extend([
            dict(shape=shape_3d, dim=[1]),
            dict(shape=shape_3d, dim=[-1, 0], keepdim=True),
        ])
    if 4 in shape_by_rank:
        dim_configs.append(dict(shape=shape_by_rank[4], dim=[3, 1]))

    dim_cases = list(product(dim_configs, dtypes))

    large_integer_cases = []
    if 1 in shape_by_rank:
        integer_shape = (max(2, shape_by_rank[1][0]),)
        for dtype in dtypes:
            if dtype in ('int32', 'int64'):
                large_integer_cases.append((dict(shape=integer_shape, large_integer=True), dtype))

    def _make_sum_input(self, case, dtype):
        shape = case['shape']
        source_shape = (*shape[:-1], shape[-1] * 2) if case.get('strided', False) else shape
        tensor = self._generate_tensor(source_shape, dtype, min=-4, max=4)
        if case.get('large_integer', False):
            large_value = {'int32': 2**24, 'int64': 2**40}[dtype]
            tensor.zero_()
            tensor[0] = large_value + 1
            tensor[1] = -large_value
        return tensor

    @parametrize('case,dtype', dim_cases + large_integer_cases)
    def test_aten_sum_dim_IntList(self, case, dtype):
        shape = case['shape']
        dim = case.get('dim', [-1])
        keepdim = case.get('keepdim', False)
        result_dtype = case.get('result_dtype')
        strided = case.get('strided', False)

        def op_calc(tensor, dim, keepdim, result_dtype, strided):
            if strided:
                tensor = tensor[..., ::2]
            y = torch.ops.aten.sum.dim_IntList(tensor, dim, keepdim, dtype=result_dtype)
            return y

        def op_calc_1(tensor, dim, keepdim, result_dtype, strided):
            if strided:
                tensor = tensor[..., ::2]
            y = torch.ops.aten.sum.dim_IntList(tensor, dim, keepdim, dtype=result_dtype)
            return y + y

        tensor = self._make_sum_input(case, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, dim, keepdim, result_dtype, strided)
```
