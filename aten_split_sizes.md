# aten.split.sizes

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.split.sizes(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]
```
### 1.2 算子功能
`aten.split.sizes` 将输入张量沿指定维度 `dim` 按**给定的各块大小列表** `split_sizes` 拆分成多个子张量，返回子张量列表（`Tensor[]`）：

1. **按列表拆分**：第 i 块在 `dim` 维度上的大小为 `split_sizes[i]`，各块依次拼接后等于原张量。
2. **约束条件**：`sum(split_sizes)` 必须等于 `self.size(dim)`，否则报错。
3. **overload 关系**：等价于 `aten.split_with_sizes`（旧入口名，官方推荐使用 `split_with_sizes`）。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `split_sizes` | `SymInt[]` | 每一块沿 `dim` 维度的大小列表，总和需等于 `self.size(dim)` |
| `dim` | `int` | 拆分的维度，默认为 `0` |
| 返回值 | `Tensor[]` | 拆分后的子张量列表 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、shape 维度范围：[1-4]

2、输出子张量在 `dim` 维度上的大小依次为 `split_sizes[i]`，其他维度与 `self` 相同

### 2.3 特殊限制说明

> 无

### 2.4 使用方法

```python
    dtypes = ['int8','int32','int64','float16', 'bfloat16', 'float32','bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_aten_split_sizes(self, shape, dtype):
        def op_calc(tensor, sizes):
            y = torch.ops.aten.split.sizes(tensor, sizes , dim = 0)
            return y[0]
        def op_calc_1(tensor, sizes):
            y = torch.ops.aten.split.sizes(tensor, sizes , dim = 0)
            return y[0] + 1
        sizes = [int(shape[0]/4)]*4
        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, sizes)
```
