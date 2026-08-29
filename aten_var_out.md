# aten.var.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.var.out(
    self: Tensor, 
    dim: int[1]?, 
    correction: int?, 
    keepdim: bool = False, 
    *, 
    out: Tensor
) -> Tensor
```
### 1.2 算子功能
`aten.var.out` 是 `aten.var` 的 out 变体（out variant）：在指定维度 `dim` 上计算输入张量 `self` 的方差，并将计算结果直接覆盖写入到预先指定的 `out` 张量中。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `dim` | `int[1]?` | 待归约的维度，可以是单个维度或维度列表；为 None 时对所有维度归约 |
| `correction` | `int?` | 样本大小与样本自由度之间的差值（校正项），为 None 时默认为 1 |
| `keepdim` | `bool` | 输出张量是否保留 `dim` 维度（大小为 1），默认为 False |
| `out` | `Tensor` | 输出张量，其数据类型（Dtype）和设备（Device）必须与输入张量匹配，形状必须与预期输出严格一致 |
| 返回值 | `Tensor` | `out` 张量本身 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × |
| NPU | - | × | - | - | - | × | - | × | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`dim` 的取值必须在 `[-self.dim(), self.dim())` 范围内

### 2.3 特殊限制说明

> 仅支持浮点类型（fp16、fp32、bf16）。当 `keepdim=False` 时输出会去掉被归约的维度；`N - correction` 必须大于 0。

### 2.4 使用方法

```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_var_out(self, shape, dtype):
        def op_calc(input, out_t):
            torch.ops.aten.var.out(input, dim=0, unbiased=True, keepdim = False, out = out_t)
            return out_t
        def op_calc_1(input, out_t):
            temp = torch.empty_like(out_t)
            torch.ops.aten.var.out(input, dim=0, unbiased=True, keepdim = False, out = temp)
            return temp + temp
        input = self._generate_tensor(shape, dtype)
        out_shape = list(shape)
        out_shape.pop(0)
        out_t = self._generate_tensor(tuple(out_shape), dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input, out_t)
```
