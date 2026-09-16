# aten.max.names_dim_max

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.max.names_dim_max(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor values, Tensor indices)
```

### 1.2 算子功能

对指定维度或全部元素取最大值。同时返回索引。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 带维度名称的输入。 |
| `dim` | `Dimname` | 归约维度名称。 |
| `keepdim` | `bool` | 是否保留归约维度。 |
| `max` | `Tensor` | 最大值输出。 |
| `max_values` | `Tensor` | 索引输出。 |
| `返回值` | `(Tensor, Tensor)` | 最大值及索引。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | x | x | x | - |
| NPU | - | x | - | - | - | x | - | x | x | x | x | - |

#### 2.2.2 Shape 支持

1. 本测试覆盖的 shape 维度范围为 `[1, 4]`。
2. 测试 shape：`(4,)`、`(4, 8)`、`(4, 8, 16)`、`(4, 8, 16, 32)`。

### 2.3 特殊限制说明

报错aten::new_empty is not yet supported with named tensors. Please drop names via `tensor = tensor.rename(None)`与上游一致，Named Tensor在官方不成熟，inductor当前不支持。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    dims = [0, -1]
    keepdims = [False, True]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    @parametrize('dim', dims)
    @parametrize('keepdim', keepdims)
    def test_aten_max_names_dim_max(self, shape, dtype, dim, keepdim):

        def op_calc(first_element, dim, keepdim, out, indices):
            y = torch.ops.aten.max.names_dim_max(first_element, dim, keepdim, max=out, max_values=indices)
            return (y[0].rename(None), y[1].rename(None))

        def op_calc_1(first_element, dim, keepdim, out, indices):
            y = torch.ops.aten.max.names_dim_max(first_element, dim, keepdim, max=out, max_values=indices)
            y = (y[0].rename(None), y[1].rename(None))
            return (y[0] + y[0], *y[1:])
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        names = ('N', 'C', 'H', 'W')[:len(shape)]
        tensor1 = tensor1.refine_names(*names)
        dim = names[dim]
        axis = names.index(dim)
        output_shape = tuple(1 if i == axis else s for i, s in enumerate(shape)) if keepdim else tuple(s for i, s in enumerate(shape) if i != axis)
        out = self._generate_tensor(output_shape, dtype, min=-2, max=2)
        indices = self._generate_tensor(output_shape, 'int64', min=0, max=1)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, dim, keepdim, out, indices,
        )
```
