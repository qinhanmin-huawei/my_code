# aten.squeeze_copy.dim

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.squeeze_copy.dim(Tensor self, int dim) -> Tensor
```

### 1.2 算子功能

删除指定 `dim` 轴上长度为 1 的维度，返回独立拷贝，元素值不变。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `dim` | `int` | 待压缩的轴，支持负索引 |
| 返回值 | `Tensor` | 保持 self 的 dtype 和设备的独立结果张量 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、`self` 的维度范围：[1-4]，输出可为 0 维。

2、指定维度长度为 1 时删除该维度，否则 shape 不变。

### 2.3 特殊限制说明

> `dim` 范围为 `[-self.ndim, self.ndim - 1]`，越界时报错。 即使 shape 不变，结果也不与输入共享存储。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    dim_cases = [(shape, len(shape) // 2) for shape in shapes]

    # Derive extras from one framework shape per rank.
    shape_by_rank = {}
    for shape in shapes:
        shape_by_rank.setdefault(len(shape), tuple(shape))

    for rank, shape in shape_by_rank.items():
        if rank == 1:
            dim_cases.append(((1,), 0))
        elif rank == 3:
            dim_cases.extend([
                ((shape[0], 1, shape[2]), 1),  # Middle axis.
                ((shape[0], shape[1], 1), -1),  # Negative axis.
            ])
            if shape[1] != 1:  # A selected non-singleton axis stays unchanged.
                dim_cases.append(((1, shape[1], 1), 1))

    @parametrize('shape,dim', dim_cases)
    @parametrize('dtype', dtypes)
    def test_aten_squeeze_copy_dim(self, shape, dtype, dim):
        def op_calc(tensor, dim):
            y = torch.ops.aten.squeeze_copy.dim(tensor, dim)
            return y

        def op_calc_1(tensor, dim):
            y = torch.ops.aten.squeeze_copy.dim(tensor, dim)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, dim)
```
