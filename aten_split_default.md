# aten.split.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.split.default(self: Tensor, split_size: int, dim: int = 0) -> Tensor[]
```
### 1.2 算子功能
`aten.split.default` 是 `aten.split` 的默认重载（default overload），其功能与 `aten.split` 一致：将输入张量 `self` 沿指定维度 `dim` 按固定大小 `split_size` 切分为多个子张量，返回一个张量元组。每个子张量的大小不超过 `split_size`；若 `self.shape[dim]` 不能被 `split_size` 整除，则最后一个子张量较小。所有子张量均为原张量的视图。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 源张量 |
| `split_size` | `int` | 单个分块在 `dim` 维度上的大小 |
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


### 2.3 特殊限制说明

> `split_size` 必须为大于 0 的整数；若大于 `self.shape[dim]`，则只返回包含整个张量在内的单一块。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_split_default(self, shape, dtype):
        def op_calc(input):
            y = torch.ops.aten.split.default(input, [tensor.shape[0]//2,tensor.shape[0]-tensor.shape[0]//2], dim = 0)
            return y[0]
        def op_calc_1(input):
            y = torch.ops.aten.split.default(input, [tensor.shape[0]//2,tensor.shape[0]-tensor.shape[0]//2], dim = 0)
            return y[0] + 1
        input = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input)
```
