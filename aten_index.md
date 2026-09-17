# aten.index

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.index(Tensor self, Tensor?[] indices) -> Tensor
```
### 1.2 算子功能
1、`aten.index` 是张量索引的 packet 调用形式，按实参解析到 `aten.index.Tensor`。

2、按 `indices` 逐维取值，输出 dtype 恒等于 `self` 的 dtype。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 被索引的源张量 |
| `indices` | `List[Optional[Tensor]]` | 索引列表，长度 = `self.dim()`，元素为 int64 索引张量或 `None` |
| 返回值 | `Tensor` | 索引结果，dtype 与 `self` 相同 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

> 表中为 `self` 的 dtype；输出 dtype **恒等于 `self` 的 dtype**，`indices` 不参与 dtype 判定（须为 int64 或 bool mask）。

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、结果形状 = 被索引维按索引长度替换、其余维原样保留（例：只索引第 0 维时为 `(index_len,) + shape[1:]`，1D 时 `indices` 只有一项，4D 时为 `[idx, None, None, None]`）。


### 2.3 特殊限制说明

> 1、`indices` 的长度必须等于 `self.dim()`，元素为 int64 索引张量或 `None`（该维全取）；索引值不得越界（CANN 的 `aclnnIndex` 明确“索引数据不支持越界”）。

> 2、输出 dtype 恒等于 `self` 的 dtype，只搬值不做类型转换。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_index(self, shape, dtype):
        # indices 只索引第 0 维，其余维填 None（全取）
        n_none = len(shape) - 1

        def op_calc(first_element, index_0):
            y = torch.ops.aten.index(first_element, [index_0] + [None] * n_none)
            return y

        def op_calc_1(first_element, index_0):
            y = torch.ops.aten.index(first_element, [index_0] + [None] * n_none)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        # 索引长度 3、取值互不相同；shape[0] 在本用例 4 组 shape 里恒为 4，不会越界
        index_0 = torch.arange(3, dtype=torch.int64, device="npu")
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, index_0)
```
