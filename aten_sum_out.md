# aten.sum.out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sum.out(
    Tensor self,
    *,
    ScalarType? dtype=None,
    Tensor(a!) out
) -> Tensor(a!)
```

### 1.2 算子功能

对全部元素求和，结果写入 `out`：`out = sum(self)`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `dtype` | `ScalarType?` | 可选的归约结果类型；指定时输入按该类型参与求和 |
| `out` | `Tensor(a!)` | 独立分配的输出张量；当前规格要求与归约结果同 shape、同 dtype，与 self 同设备 |
| 返回值 | `Tensor(a!)` | 写入结果的 out，与传入 out 别名 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

`out` 默认配置：浮点输入使用输入 dtype，整数和 bool 输入使用 int64；显式指定 fp32 时，`out` 也使用 fp32。

#### 2.2.2 Shape 支持

1、`self` 支持的维度范围：[1-4]。

2、对全部维度求和，输出为零维张量。

3、支持非整齐尺寸、步长为 2 的非连续输入和空归约；空归约的结果为零。

### 2.3 特殊限制说明

> 当前规格要求 `out` 的 shape 与归约结果一致；显式 `dtype` 必须与 `out.dtype` 一致。当前环境下，`dtype=None` 时，`out` 须匹配输入的默认归约结果类型。

### 2.4 使用方法

示例需先执行 `from itertools import product`；`_make_sum_input` 统一准备普通、步长布局及大整数用例的输入。

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    shape_by_rank = {}
    for shape in shapes:
        shape_by_rank.setdefault(len(shape), tuple(shape))

    sum_configs = [dict(shape=shape) for shape in shapes]
    if shapes:
        sum_configs.append(dict(shape=shapes[0], result_dtype=torch.float32))

    if 2 in shape_by_rank:
        shape_2d = shape_by_rank[2]
        irregular_shape = (shape_2d[0] + 1, max(1, shape_2d[1] - 1))
        empty_shape = (shape_2d[0], 0)
        sum_configs.extend([
            dict(shape=irregular_shape),
            dict(shape=empty_shape),
        ])

        sum_configs.append(dict(shape=shape_2d, strided=True))

    sum_cases = list(product(sum_configs, dtypes))

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

    @parametrize('case,dtype', sum_cases + large_integer_cases)
    def test_aten_sum_out(self, case, dtype):
        shape = case['shape']
        result_dtype = case.get('result_dtype')
        strided = case.get('strided', False)

        def op_calc(tensor, result_dtype, strided, out):
            if strided:
                tensor = tensor[..., ::2]
            y = torch.ops.aten.sum.out(tensor, dtype=result_dtype, out=out)
            return y

        def op_calc_1(tensor, result_dtype, strided, out):
            if strided:
                tensor = tensor[..., ::2]
            y = torch.ops.aten.sum.out(tensor, dtype=result_dtype, out=out)
            return y + y

        tensor = self._make_sum_input(case, dtype)
        out_shape = ()
        out_dtype = result_dtype
        if out_dtype is None:
            out_dtype = tensor.dtype if tensor.is_floating_point() else torch.int64
        out = torch.empty(out_shape, dtype=out_dtype, device=tensor.device)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, result_dtype, strided, out)
```
