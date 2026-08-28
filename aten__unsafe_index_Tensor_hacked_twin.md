# aten._unsafe_index.Tensor_hacked_twin

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten._unsafe_index.Tensor_hacked_twin(
    Tensor self, 
    Tensor?[] index
) -> (Tensor, Tensor)
```
### 1.2 算子功能
`aten._unsafe_index.Tensor_hacked_twin` 是高级索引算子的 **NPU hacked twin 变体**（torch_npu 扩展提供）。执行与 `aten.index.Tensor` 相同的高级索引操作，并额外返回一个 twin 输出张量：

1. **高级索引**：使用索引张量列表 `index`（其中 `None` 表示跳过该维度，等价于切片 `:`）从 `self` 中选取元素，结果形状由广播后的索引张量与非索引维度共同决定。例如 `x[row, col]`，其中 `row`、`col` 为等长的 int 型索引张量，结果形状为 `(len(row), len(col))` 与非索引维度的组合。
2. **unsafe 语义**：跳过公开版 `index` 的形状合法性检查与边界校验，要求调用方保证索引合法，否则行为未定义。
3. **twin 输出**：返回值包含两个张量 `(y, twin)`，第 1 个为索引结果，第 2 个为 twin 输出（内部用于加速/profiling 场景，一般直接消费 `y`）。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 被索引的数据张量 |
| `index` | `Tensor?[]` | 各维度的索引张量列表（int 型），`None` 表示跳过该维度；有效索引张量之间按广播规则处理 |
| 返回值 1 | `Tensor` | 高级索引的结果张量 |
| 返回值 2 | `Tensor` | twin 输出张量 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU |  |  |  |  |  |  |  |  |  |  |  |  |
| NPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ |  |

#### 2.2.2 Shape 支持

1、索引结果形状 = 广播后的索引形状 ⊕ 未索引维度形状

2、当前测试覆盖：`self` 为 `[2, 3]` 的 float16 张量，索引为 `[row, col]`（`row`/`col` 为长度 2 的 int32 张量），结果形状 `[2, 2]`

### 2.3 特殊限制说明

1. 该算子为 **torch_npu 扩展算子**（hacked twin 变体），GPU 不存在此 overload。
2. **unsafe 语义**：不做越界/形状校验，索引必须保证合法，否则未定义行为。
3. 当前数据类型支持仅以实际验证覆盖为准（float16 数据 + int32 索引）。

### 2.4 使用方法

```python
    dtypes = ['int8','int32','int64','float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]
    def test_aten_unsafe_index_tensor_hacked_twin(self):
        def op_calc(x, row, col):
            y, twin = torch.ops.aten._unsafe_index.Tensor_hacked_twin(x, [row, col])
            result = y
            return result
        def op_calc_1(x, row, col):
            y, twin = torch.ops.aten._unsafe_index.Tensor_hacked_twin(x, [row, col])
            result = y + y
            return result
        x = torch.randn([2, 3], dtype=torch.float16, device=torch.device("npu"))
        row = torch.tensor([0, 1], dtype=torch.int32, device=torch.device("npu"))
        col = torch.tensor([2, 0], dtype=torch.int32, device=torch.device("npu"))
        self.execute_inductor_test(None, None, op_calc, op_calc_1, x, row, col)
```
