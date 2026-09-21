# aten.unsqueeze

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
```

本文记录 Tensor 调用入口，对应 `aten.unsqueeze.default`。

### 1.2 算子功能

在 `dim` 位置插入长度为 1 的维度，返回与 `self` 共享存储的视图，元素值不变。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor(a)` | 输入张量 |
| `dim` | `int` | 新维度的插入位置，支持负索引 |
| 返回值 | `Tensor(a)` | 比输入多一维，保持输入 dtype 和设备 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、`self` 的维度范围：[0-4]，输出维度范围：[1-5]。

2、零维输入增加维度后，输出 shape 为 `(1,)`。

### 2.3 特殊限制说明

> `dim` 范围为 `[-self.ndim - 1, self.ndim]`；`dim=self.ndim` 或 `dim=-1` 表示在末尾插入。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    cases = [dict(shape=shape, dim=0) for shape in shapes]
    shape_by_rank = {}
    for shape in shapes:
        shape_by_rank.setdefault(len(shape), tuple(shape))

    if 1 in shape_by_rank:
        cases.extend([
            dict(shape=(), dim=0),
            dict(shape=(), dim=-1),
        ])
    if 2 in shape_by_rank:
        shape = shape_by_rank[2]
        cases.extend([
            dict(shape=shape, dim=len(shape)),
            dict(shape=shape, dim=-1),
            dict(shape=shape, dim=-len(shape) - 1),
            dict(shape=shape, dim=1, strided=True),
        ])
    if 3 in shape_by_rank:
        cases.append(dict(shape=shape_by_rank[3], dim=1))

    @parametrize('case', cases)
    @parametrize('dtype', dtypes)
    def test_aten_unsqueeze(self, case, dtype):
        shape = case['shape']
        dim = case['dim']
        strided = case.get('strided', False)

        def op_calc(tensor, dim, strided):
            if strided:
                tensor = tensor[..., ::2]
            y = torch.ops.aten.unsqueeze(tensor, dim)
            return y

        def op_calc_1(tensor, dim, strided):
            if strided:
                tensor = tensor[..., ::2]
            y = torch.ops.aten.unsqueeze(tensor, dim)
            return y + y

        source_shape = (*shape[:-1], shape[-1] * 2) if strided else shape
        tensor = self._generate_tensor(source_shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, dim, strided)
```
