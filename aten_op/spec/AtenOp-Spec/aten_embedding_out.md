# aten.embedding.out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.embedding.out(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False, *, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能

按 indices 从二维权重表读取对应行。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `weight` | `Tensor` | 权重张量。 |
| `indices` | `Tensor` | 索引张量或索引输出缓冲区。 |
| `padding_idx` | `int` | 算子配置参数，取值见使用方法。 |
| `scale_grad_by_freq` | `bool` | 算子配置参数，取值见使用方法。 |
| `sparse` | `bool` | 算子配置参数，取值见使用方法。 |
| `out` | `Tensor` | 输出缓冲区。 |
| `返回值` | `Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |

#### 2.2.2 Shape 支持

`weight` 固定为 2 维 Tensor；`indices` 支持 1～4 维 Tensor。

### 2.3 特殊限制说明

weight 的 shape 固定为 `(128, 128)`；dtype 表描述 weight；indices 仅使用 int32/int64，范围为 `[0, 128)`。仅覆盖前向。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    index_dtypes = ['int32', 'int64']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    @parametrize('index_dtype', index_dtypes)
    def test_aten_embedding_out(self, shape, dtype, index_dtype):

        def op_calc(weight, indices, padding_idx, scale_grad_by_freq, sparse, out):
            return torch.ops.aten.embedding.out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out=out)

        def op_calc_1(weight, indices, padding_idx, scale_grad_by_freq, sparse, out):
            y = torch.ops.aten.embedding.out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out=out)
            return y + y
        weight = self._generate_tensor((128, 128), dtype, min=-2, max=2)
        indices = self._generate_tensor(shape, index_dtype, min=0, max=128)
        padding_idx = -1
        scale_grad_by_freq = False
        sparse = False
        out = self._generate_tensor(tuple(shape) + (128,), dtype, min=-2, max=2)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, weight, indices, padding_idx, scale_grad_by_freq, sparse, out,
        )
```
