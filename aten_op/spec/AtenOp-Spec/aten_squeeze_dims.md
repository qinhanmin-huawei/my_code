# aten.squeeze.dims

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)
```

### 1.2 算子功能

删除 `dim` 列表指定且长度为 1 的维度，返回与 `self` 共享存储的视图，元素值不变。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor(a)` | 输入张量 |
| `dim` | `int[]` | 待压缩的轴列表，按原输入轴编号解释 |
| 返回值 | `Tensor(a)` | 保持 self 的 dtype 和设备的视图 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、`self` 的维度范围：[1-4]，输出可为 0 维。

2、只删除指定且长度为 1 的维度；空列表不删除任何维度。

### 2.3 特殊限制说明

> 各维度须在合法范围内，支持负索引；归一化后不允许重复，同一列表中的轴均按原输入编号解释。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    dims_cases = [dict(shape=shape, dims=[0] if len(shape) == 1 else [0, -1]) for shape in shapes]

    shape_by_rank = {}
    for shape in shapes:
        shape_by_rank.setdefault(len(shape), tuple(shape))

    for rank, shape in shape_by_rank.items():
        if rank == 2:
            dims_cases.append(dict(shape=(1, 1), dims=[0, 1]))
        elif rank == 3:
            dims_cases.append(dict(shape=(1, shape[1], 1), dims=[]))
        elif rank == 4:
            dims_cases.append(dict(shape=(shape[0], 1, shape[2], 1), dims=[3, 1]))
            if shape[1] != 1:
                dims_cases.append(dict(shape=(1, shape[1], 1, shape[3]), dims=[-2, 1]))

    @parametrize('case', dims_cases)
    @parametrize('dtype', dtypes)
    def test_aten_squeeze_dims(self, case, dtype):
        shape = case['shape']
        dims = case['dims']

        def op_calc(tensor, dims):
            y = torch.ops.aten.squeeze.dims(tensor, dims)
            return y

        def op_calc_1(tensor, dims):
            y = torch.ops.aten.squeeze.dims(tensor, dims)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, dims)
```
