# aten.var_mean.dim

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.var_mean.dim(
    Tensor self,
    int[1]? dim,
    bool unbiased=True,
    bool keepdim=False
) -> (Tensor, Tensor)
```

### 1.2 算子功能

沿指定维度计算方差和均值，依次返回 `(var, mean)`：`var = sum((x - mean(x))**2) / max(0, N - c)`。N 为每组归约元素数，c 在 unbiased=True 时为 1，否则为 0。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `dim` | `int[1]?` | 归约维度；支持负轴和多轴，None 或空列表表示全部维度 |
| `unbiased` | `bool` | True 时除以 N−1，False 时除以 N；默认 True |
| `keepdim` | `bool` | 是否将被归约维度保留为 1，默认 False |
| 返回值 | `(Tensor, Tensor)` | 依次为方差和均值，当前浮点规格下均与输入同 dtype |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self | - | × | - | - | - | × | - | × | √ | √ | √ | × |
| NPU | self | - | × | - | - | - | × | - | × | √ | √ | √ | × |

int8、int32、int64 和 bool 输入社区不支持，当前 CPU Eager 和 NPU Eager 均不支持；无可用 Eager 精度参考，未测试对应 Inductor 路径。

#### 2.2.2 Shape 支持

1、`self` 的维度范围：[1-4]。

2、`keepdim=False` 时删除被归约维度，True 时将其长度置为 1；全量归约且不保留维度时输出为零维张量。

### 2.3 特殊限制说明

> `dim` 必须有效且不能重复。空归约的方差和均值均为 NaN；N=1 且输入为有限值时，方差在 `unbiased=True` 下为 NaN，在 False 下为 0。均值不受 `unbiased` 影响。

### 2.4 使用方法

```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    keepdims = [False, True]
    unbiaseds = [False, True]
    dim_cases = [dict(shape=shape, dim=[dim]) for shape in shapes for dim in (0, -1)]
    multi_dim_shapes = [shape for shape in shapes if len(shape) == 3][:1]
    dim_cases.extend(dict(shape=shape, dim=[0, -1]) for shape in multi_dim_shapes)

    @parametrize('case', dim_cases)
    @parametrize('dtype', dtypes)
    @parametrize('keepdim', keepdims)
    @parametrize('unbiased', unbiaseds)
    def test_aten_var_mean_dim(self, case, dtype, keepdim, unbiased):
        shape = case['shape']
        dim = case['dim']

        def op_calc(tensor, dim, unbiased, keepdim):
            y, mean = torch.ops.aten.var_mean.dim(tensor, dim, unbiased, keepdim)
            return y, mean

        def op_calc_1(tensor, dim, unbiased, keepdim):
            y, mean = torch.ops.aten.var_mean.dim(tensor, dim, unbiased, keepdim)
            return y + y, mean + mean

        tensor = self._generate_tensor(shape, dtype, min=-4, max=4)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, dim, unbiased, keepdim)
```
