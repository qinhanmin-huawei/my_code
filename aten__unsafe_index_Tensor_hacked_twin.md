# aten._unsafe_index.Tensor_hacked_twin

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten._unsafe_index.Tensor_hacked_twin(
    Tensor self, 
    Tensor[] indices
) -> Tensor
```
### 1.2 算子功能
`aten._unsafe_index.Tensor_hacked_twin` 是高级索引算子的 **NPU hacked twin 变体**（torch_npu 扩展提供）。执行与 `aten.index.Tensor` 相同的高级索引操作，并额外返回一个 twin 输出张量：

1. **unsafe 语义**：它是标准 aten::index 的优化版本。它在底层跳过了对索引是否越界（Out-of-Bounds）的运行时检查。这大幅降低了 CPU/GPU 之间的同步开销，通常由编译器（如 torch.compile）在确认索引数据在数学上绝对安全时（例如双线性插值、上采样等内部生成的索引）自动生成。
2. **Hacked Twin**：标准的 PyTorch 索引允许列表内混入 None（如 [None, index_tensor]）。为了让底层图编译器（如 Torch-MLIR、昇腾 CANN 等）能够进行静态类型推导，该算子通过内部手段（Hack）将输入规范化，强制要求索引列表必须是纯粹的张量数组 Tensor[]，移除了所有 None 元素。
## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 被索引的数据张量 |
| `indices` | `Tensor[]` | 索引张量列表。一个由整型张量（通常为 int32 或 int64）组成的列表，用于指定要在 self 的哪些维度、哪些位置提取数据。列表中不能包含 None。 |
| 返回值  | `Tensor` | 结果张量。包含根据 indices 过滤、重组后的新数据。其形状（Shape）由 self 的未索引维度与 indices 内部张量的广播形状共同决定。 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU |  |  |  |  |  |  |  |  |  |  |  |  |
| NPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、索引结果形状 = 广播后的索引形状 ⊕ 未索引维度形状


### 2.3 特殊限制说明

1. 该算子为 **torch_npu 扩展算子**（hacked twin 变体），GPU 不存在此 overload。
2. **unsafe 语义**：不做越界/形状校验，索引必须保证合法，否则未定义行为。

### 2.4 使用方法

```python
    dtypes = ['int8','int32','int64','float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_aten_unsafe_index_tensor_hacked_twin(self, shape, dtype):
        def op_calc(x, indices):
            return torch.ops.aten._unsafe_index.Tensor_hacked_twin(x, indices)
        def op_calc_1(x, indices):
            y = torch.ops.aten._unsafe_index.Tensor_hacked_twin(x, indices)
            return y + y
        x = self._generate_tensor(shape, dtype)
        indices = []
        for dim_size in shape:
            idx = torch.tensor([0, dim_size - 1], dtype=torch.int32, device=x.device)
            indices.append(idx)

        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, x, indices)
```
