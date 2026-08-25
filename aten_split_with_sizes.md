# aten.split_with_sizes

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.split_with_sizes(self: Tensor, split_sizes: int[], dim: int = 0) -> Tensor[]
```
### 1.2 算子功能
`aten.split_with_sizes` 将输入张量沿指定维度 `dim` 切分为多个子张量，每个子张量在 `dim` 维度上的大小由 `split_sizes` 列表中的对应元素显式指定。返回一个张量元组，元组长度等于 `split_sizes` 的长度。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 源张量 |
| `split_sizes` | `int[]` | 各切分块在 `dim` 维度上的大小列表 |
| `dim` | `int` | 切分维度，默认为 0 |
| 返回值 | `Tensor[]` | 切分后的子张量元组，与原张量共享存储 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`split_sizes` 各元素之和必须等于 `self.shape[dim]`

### 2.3 特殊限制说明

> 无

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_split_with_sizes_default(self, shape, dtype):
        def op_calc(input, split_sizes):
            y = torch.ops.aten.split_with_sizes(input, split_sizes, dim=0)
            return y[0]
        def op_calc_1(input, split_sizes):
            y = torch.ops.aten.split_with_sizes(input, split_sizes, dim=0)
            return y[0] + 1
        split_sizes = [int(shape[0]/4)]*4    
        input = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input, split_sizes)
```