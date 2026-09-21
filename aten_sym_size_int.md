# aten.sym_size.int

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sym_size.int(Tensor self, int dim) -> SymInt
```

### 1.2 算子功能

返回指定轴的长度：`y = self.shape[dim]`，不读取元素值。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `dim` | `int` | 查询轴，支持负索引 |
| 返回值 | `SymInt` | 轴长度；具体形状下表现为整数，符号形状下可保留符号表达式 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、`self` 的维度范围：[1-4]。

2、轴长度允许为 0 或 1。

3、支持二维连续输入的动态轴长度查询。

### 2.3 特殊限制说明

> `dim` 范围为 `[-self.ndim, self.ndim - 1]`。静态查询可被常量化；查询轴长度为动态值时，可保留为运行时参数。

### 2.4 使用方法

示例用 int64 Tensor 承载查询值，算子本身返回 `SymInt`；示例计时包含结果写入和后续运算，不代表查询算子的独立 kernel 性能。

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    cases = [dict(shape=shape, dim=0) for shape in shapes]
    shape_by_rank = {}
    for shape in shapes:
        shape_by_rank.setdefault(len(shape), tuple(shape))

    if 1 in shape_by_rank:
        cases.append(dict(shape=(1,), dim=0))
    if 2 in shape_by_rank:
        shape = shape_by_rank[2]
        cases.extend([
            dict(shape=shape, dim=-1),
            dict(shape=shape, dim=-1, strided=True),
            dict(shape=(shape[0], 0), dim=-1),
            dict(shape=shape, dim=1, dynamic=True),
        ])
    if 3 in shape_by_rank:
        cases.append(dict(shape=shape_by_rank[3], dim=1))

    @parametrize('case', cases)
    @parametrize('dtype', dtypes)
    def test_aten_sym_size_int(self, case, dtype):
        shape = case['shape']
        dim = case['dim']
        strided = case.get('strided', False)
        dynamic = case.get('dynamic', False)

        def op_calc(tensor, dim, strided):
            if strided:
                tensor = tensor[..., ::2]
            size = torch.ops.aten.sym_size.int(tensor, dim)
            y = torch.full((1,), size, dtype=torch.int64, device=tensor.device)
            return y

        def op_calc_1(tensor, dim, strided):
            if strided:
                tensor = tensor[..., ::2]
            size = torch.ops.aten.sym_size.int(tensor, dim)
            y = torch.full((1,), size, dtype=torch.int64, device=tensor.device)
            return y + y

        shapes = [shape, (shape[0], shape[1] + 8)] if dynamic else [shape]
        for shape in shapes:
            source_shape = (*shape[:-1], shape[-1] * 2) if strided else shape
            tensor = self._generate_tensor(source_shape, dtype)
            if dynamic:
                torch._dynamo.mark_dynamic(tensor, 1)
            self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, dim, strided)
```
