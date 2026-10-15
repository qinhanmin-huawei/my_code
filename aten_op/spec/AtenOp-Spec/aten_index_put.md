# aten.index_put

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
```
### 1.2 算子功能
1、`aten.index_put` 是 packet 调用形式，解析到 `aten.index_put.default`：把 `values` 写入 `self` 中 `indices` 指定的位置，返回写入后的新张量。

2、`accumulate=False` 为覆盖写，`True` 为累加写（等价于 `index_add` / `scatter_add` 的语义）。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 被写入的源张量 |
| `indices` | `List[Optional[Tensor]]` | 索引列表，长度 = `self.dim()`，元素为 int64 索引张量或 `None` |
| `values` | `Tensor` | 写入的值，形状为 `(index_len,) + shape[1:]`，dtype 必须与 `self` 相同 |
| `accumulate` | `bool` | 默认 `False`（覆盖写），`True` 为累加写 |
| 返回值 | `Tensor` | 写入后的新张量，dtype 与 `self` 相同 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

`accumulate=False`（覆盖写）：

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

`accumulate=True`（累加写，走 `atomic_add`）：

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | ×    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | ×    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

> 表中为 `self` 与 `values` 的 dtype（两者必须一致）；输出 dtype **恒等于 `self` 的 dtype**。

> `accumulate=True` 走原子累加，int8 在 GPU/NPU 上均无 `atomic_add` 实现。

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`indices` 的长度等于 `self.dim()`；`values` 的形状必须等于被索引区域，只索引第 0 维时为 `(index_len,) + shape[1:]`。


### 2.3 特殊限制说明

1、 `values` 与 `self` 必须同 dtype，没有提升/回退。

2、 `accumulate=False` 时重复索引的写入顺序在 PyTorch 里是未定义的，用例的索引值互不相同。

3、 `accumulate=True` 走原子累加：int8 在 GPU/NPU 上都不支持，`atomic_add does not support int8`，编译报错。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    accumulates = [False, True]
    # (dtype, accumulate) 组合轴：int8 × accumulate=True 已摘除（atomic_add 不支持 int8）
    dtype_accumulate = [(d, a) for d in dtypes for a in accumulates
                        if not (d == 'int8' and a)]

    @parametrize('shape', shapes)
    @parametrize('dtype,accumulate', dtype_accumulate)
    def test_aten_index_put(self, shape, dtype, accumulate):
        # indices 只索引第 0 维，其余维填 None（全取）
        n_none = len(shape) - 1

        def op_calc(first_element, index_0, values):
            y = torch.ops.aten.index_put(first_element, [index_0] + [None] * n_none,
                                         values, accumulate)
            return y

        def op_calc_1(first_element, index_0, values):
            y = torch.ops.aten.index_put(first_element, [index_0] + [None] * n_none,
                                         values, accumulate)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        # 索引长度 3、取值互不相同，且不越界（shape[0] 在本文 4 组 shape 里恒为 4）
        index_0 = torch.arange(min(3, shape[0]), dtype=torch.int64, device="npu")
        # values 的形状 = 被索引区域的形状，dtype 必须与 self 相同
        values = self._generate_tensor((index_0.numel(),) + tuple(shape[1:]), dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   first_element, index_0, values)
```
