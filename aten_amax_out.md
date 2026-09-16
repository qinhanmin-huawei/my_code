# aten.amax.out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.amax.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能

对指定维度或全部元素取最大值。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `dim` | `List[int]` | 归约维度。 |
| `keepdim` | `bool` | 是否保留归约维度。 |
| `out` | `Tensor` | 输出缓冲区。 |
| `返回值` | `Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |

#### 2.2.2 Shape 支持

1. 本测试覆盖的 shape 维度范围为 `[1, 4]`。
2. 测试 shape：`(4,)`、`(4, 8)`、`(4, 8, 16)`、`(4, 8, 16, 32)`。

### 2.3 特殊限制说明

输入非空；dim 版本覆盖首维、末维及 keepdim 开关。 不覆盖复数。

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
    def test_aten_amax_out(self, shape, dtype, dim, keepdim):

        def op_calc(first_element, dim, keepdim, out):
            y = torch.ops.aten.amax.out(first_element, [dim], keepdim, out=out)
            return y

        def op_calc_1(first_element, dim, keepdim, out):
            y = torch.ops.aten.amax.out(first_element, [dim], keepdim, out=out)
            return y + y
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        axis = dim % len(shape)
        output_shape = tuple(1 if i == axis else s for i, s in enumerate(shape)) if keepdim else tuple(s for i, s in enumerate(shape) if i != axis)
        out = self._generate_tensor(output_shape, dtype, min=-2, max=2)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, dim, keepdim, out,
        )
```
