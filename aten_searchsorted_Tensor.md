# aten.searchsorted.Tensor

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.searchsorted.Tensor(
sorted_sequence: Tensor,
self: Tensor,
*,
out_int32: bool = False,
right: bool = False,
side: Optional[str] = None,
sorter: Optional[Tensor] = None
) -> Tensor
```


### 1.2 算子功能
`aten.searchsorted.Tensor` 在 `sorted_sequence` 的**最后一维**上执行二分查找，为 `values` 中每个元素找到插入索引，以保持顺序。

- 若 `right=False`（或 `side='left'`），返回**左边界**（即第一个大于等于值的索引）。
- 若 `right=True`（或 `side='right'`），返回**右边界**（即第一个大于值的索引）。
- 支持 N 维 `sorted_sequence`（在每个最后一维上分别查找）。
- 支持通过 `sorter` 提供排序索引（用于未排序的序列）。
- 返回与 `values` 形状相同的张量，dtype 由 `out_int32` 控制（`int64` 或 `int32`）。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `sorted_sequence` | `Tensor` | 在最后一维上单调递增的序列（若提供 `sorter` 则可不排序） |
| `self` / `values` | `Tensor` | 待搜索的值，形状需与 `sorted_sequence` 广播兼容 |
| `out_int32` | `bool` | 输出为 `int32`（True）或 `int64`（False），默认 `False` |
| `right` | `bool` | 边界方向：`False`=左边界（下界），`True`=右边界（上界），默认 `False` |
| `side` | `Optional[str]` | `'left'` 或 `'right'`，与 `right` 互斥；若同时设置且冲突则报错。优先级高于 `right` |
| `sorter` | `Optional[Tensor]` | 形状与 `sorted_sequence` 相同的 `torch.long` 张量，提供排序索引，用于未排序的序列 |
| 返回值 | `Tensor` | 索引张量，形状与 `values` 相同，dtype 由 `out_int32` 决定 |
### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | ×      | ×      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | ×      | ×      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64','float16', 'float32']
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    def test_aten_searchsorted_Tensor(self, shape, dtype):
        def op_calc(boundaries, tensor):
            out = torch.ops.aten.searchsorted.Tensor(
                boundaries, 
                tensor)
            return out
        def op_calc_1():
            out = torch.ops.aten.searchsorted.Tensor(
                boundaries, 
                tensor)
            return out + out
        boundaries = torch.tensor([0, 0.1, 0.4, 0.6, 0.8, 2, 4, 6, 8, 10, 40, 80, 100],device = "npu", dtype=eval("torch."+dtype))
        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, boundaries, tensor)
```
