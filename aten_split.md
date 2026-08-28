# aten.split

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.split(Tensor self, SymInt split_size, int dim=0) -> Tensor[]
```
### 1.2 算子功能
`aten.split` 将输入张量沿指定维度 `dim` 按固定块大小 `split_size` 拆分成多个子张量，返回子张量列表（`Tensor[]`）：

1. **逐块拆分**：沿 `dim` 维度按 `split_size` 大小依次切块，每块的其他维度形状与 `self` 相同。
2. **末块处理**：若 `self.size(dim)` 不能被 `split_size` 整除，最后一块的大小为剩余元素数（小于 `split_size`）。
3. **块数计算**：产生的块数为 `ceil(self.size(dim) / split_size)`；若 `self.size(dim)` 小于 `split_size`，则只产生一个块。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `split_size` | `SymInt` | 每一块沿 `dim` 维度的大小 |
| `dim` | `int` | 拆分的维度，默认为 `0` |
| 返回值 | `Tensor[]` | 拆分后的子张量列表 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ |  |
| NPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ |  |

#### 2.2.2 Shape 支持

1、shape 维度范围：[1-4]

2、输出子张量在 `dim` 维度上的大小为 `split_size`（末块可为剩余大小），其他维度与 `self` 相同

### 2.3 特殊限制说明

> 无

### 2.4 使用方法

```python
    dtypes = ['int8','int32','int64','float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_aten_split(self, shape, dtype):
        def op_calc(tensor):
            y = torch.ops.aten.split(tensor, 64 , dim = 0)
            return y[0]
        def op_calc_1(tensor):
            y = torch.ops.aten.split(tensor, 64 , dim = 0)
            return y[0] + 1
        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor)
```
