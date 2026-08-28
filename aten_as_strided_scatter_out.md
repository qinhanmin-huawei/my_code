# aten.as_strided_scatter.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.as_strided_scatter.out(
    Tensor self, 
    Tensor src, 
    SymInt[] size, 
    SymInt[] stride, 
    int? storage_offset=None, 
    *, 
    Tensor(a!) out
) -> Tensor(a!)
```
### 1.2 算子功能
`aten.as_strided_scatter.out` 以 `as_strided` 语义在 `self` 的视图上散布（scatter）`src` 的数据，并将结果写入 `out`：

1. **构造 as_strided 视图**：按 `size`、`stride`、`storage_offset` 在 `self` 上定义一个 strided 视图区域。
2. **数据散布**：将 `src` 的元素逐个复制到该视图区域对应位置，其余位置保持 `self` 原值。
3. **out 变体语义**：结果不新建张量而是写入 `out` 并返回它。整体等价于：先复制 `self` 到 `out`，再执行 `out.as_strided(size, stride, storage_offset).copy_(src)`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 源数据张量，提供背景数据与存储 |
| `src` | `Tensor` | 要散布进去的数据张量，元素数需等于视图大小 `numel(size)` |
| `size` | `SymInt[]` | 视图形状 |
| `stride` | `SymInt[]` | 视图步长，需与 `size` 匹配且不越出 `self` 存储 |
| `storage_offset` | `int?` | 视图在存储中的起始偏移量，默认 `0` |
| `out` | `Tensor(a!)` | 输出张量（原地写入），形状与 `self` 相同 |
| 返回值 | `Tensor(a!)` | 即 `out`，散布后的结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ | √ |
| NPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、shape 维度范围：[1-4]

2、输出张量形状与 `self` 相同；`src` 形状等于视图 `size`

### 2.3 特殊限制说明

1. 视图范围由 `size/stride/storage_offset` 决定，跨度不能超出 `self` 的存储空间；`src.numel()` 需与 `numel(size)` 一致。
2. 当前测试中 `new_size = (tensor.numel()//64, 64)`、`new_stride = (64, 1)`，即要求 `tensor.numel()` 可被 64 整除（视图恰好占满整个张量）。

### 2.4 使用方法

```python
    dtypes = ['int8','int32','int64','float16', 'bfloat16', 'float32','bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_aten_as_strided_scatter_out(self, shape, dtype):
        def op_calc(tensor, src_tensor, new_size, new_stride, out_t):
            out = torch.ops.aten.as_strided_scatter.out(tensor, src_tensor, new_size, new_stride, storage_offset=0, out = out_t)
            return out
        def op_calc_1(tensor, src_tensor, new_size, new_stride, out_t):
            out = torch.ops.aten.as_strided_scatter.out(tensor, src_tensor, new_size, new_stride, storage_offset=0, out = out_t)
            return out + out
        tensor = self._generate_tensor(shape, dtype)
        new_size = (tensor.numel()//64, 64)
        new_stride = (64, 1)
        src_tensor = self._generate_tensor(new_size, dtype)
        out_t = torch.zeros(new_size, dtype=eval("torch." + dtype), device="npu")
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, src_tensor, new_size, new_stride, out_t)
```
