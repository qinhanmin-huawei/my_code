# aten._unsafe_masked_index_put_accumulate.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten._unsafe_masked_index_put_accumulate.default(
    self: Tensor,
    mask: Tensor,
    indices: List[Optional[Tensor]],
    values: Tensor
) -> Tensor
```


### 1.2 算子功能
`aten._unsafe_masked_index_put_accumulate.default` 执行带**掩码（mask）** 的、**累加（accumulate）** 式的索引写入操作。其计算逻辑为：

1. 根据 `mask`（布尔张量）筛选 `values` 中需要参与累加的元素：未被掩码选中的位置用 `0` 填充。
2. 使用 `indices` 列表中的索引张量，将筛选后的值映射到目标张量 `self` 的指定位置。
3. 将映射后的值**累加**（而非覆盖）到 `self` 的对应位置。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self` | `Tensor` | 目标张量，会被原地修改并返回 |
| `mask` | `Tensor` | 布尔类型张量，形状须与 `values` 相同，用于控制哪些值参与累加 |
| `indices` | `List[Optional[Tensor]]` | 索引张量列表，长度等于 `self` 的维度数。每个元素可以是 `torch.long` 或 `torch.int` 类型的张量，或为 `None`（表示该维度全部选取） |
| `values` | `Tensor` | 包含待累加值的张量，形状须与 `indices` 广播后的形状一致 |
| 返回值 | `Tensor` | 返回修改后的 `self` 张量 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | ×    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | ×    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['int32', 'int64','float16', 'bfloat16', 'float32', "bool"]
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    def test_aten_unsafe_masked_index_put_accumulate_default(self, shape, dtype):
        values = self._generate_tensor(shape, dtype)
        input = torch.zeros_like(values)
        numel = input.numel()
        test_size = min(numel//2, 128)
        test_size = max(test_size, 1)
        test_size = 128
        perm = torch.randperm(numel, device="cpu")[:test_size].to(values.device)

        flat = perm
        indices = []
        for size in reversed(shape):
            indices.append((flat%size).to(torch.long))
            flat = flat // size
        indices = list(reversed(indices))
        indices = [torch.cat([idx,idx], dim=0) for idx in indices]

        values_sel = values.reshape(-1).index_select(0, perm)
        values_flat = torch.cat([values_sel, values_sel], dim=0)
        mask_flat = (torch.rand(values_flat.shape, device=values.device) > 0.5)
        
        def op_calc(input, mask, indices, values):
            out = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
                input, mask, indices, values
            )
            return out
        
        def op_calc_1(input, mask, indices, values):
            out = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
                input, mask, indices, values
            )
            return out + out
        
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input, mask_flat, indices, values_flat)
```
