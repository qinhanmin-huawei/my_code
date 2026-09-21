# aten.unbind.int

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]
```

### 1.2 算子功能

沿 `dim` 将输入逐片拆分，删除该维度；各输出与 `self` 共享存储，元素值不变。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor(a -> *)` | 输入张量 |
| `dim` | `int` | 拆分轴，默认 0，支持负索引 |
| 返回值 | `Tensor(a)[]` | 按索引顺序排列的切片，保持输入 dtype 和设备 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、`self` 的维度范围：[1-4]。

2、输出数量为 `self.shape[dim]`，每个输出比输入少一维；沿长度为 0 的轴拆分时返回空序列。

### 2.3 特殊限制说明

> `dim` 范围为 `[-self.ndim, self.ndim - 1]`，不接受零维输入。

### 2.4 使用方法

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
            dict(shape=(shape[0], 1), dim=1),
            dict(shape=shape, dim=-1, strided=True),
        ])
    if 3 in shape_by_rank:
        cases.append(dict(shape=shape_by_rank[3], dim=1))

    @parametrize('case', cases)
    @parametrize('dtype', dtypes)
    def test_aten_unbind_int(self, case, dtype):
        shape = case['shape']
        dim = case['dim']
        strided = case.get('strided', False)

        def op_calc(tensor, dim, strided):
            if strided:
                tensor = tensor[..., ::2]
            y = torch.ops.aten.unbind.int(tensor, dim)
            return tuple(y)

        def op_calc_1(tensor, dim, strided):
            if strided:
                tensor = tensor[..., ::2]
            y = torch.ops.aten.unbind.int(tensor, dim)
            return tuple(part + part for part in y)

        source_shape = (*shape[:-1], shape[-1] * 2) if strided else shape
        tensor = self._generate_tensor(source_shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, dim, strided)
```
