# aten.cumsum.default


## 1. OP 概述

### 1.1 算子原型
```
torch.ops.aten.cumsum.default(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
```

### 1.2 算子功能
`aten.cumsum.default` 是 `aten.cumsum` 的默认重载（default overload），其功能与 `aten.cumsum` 一致：返回输入张量 `self` 在指定维度 `dim` 上的累积和（cumulative sum）。对于每个位置，输出元素等于该位置及其之前所有元素沿指定维度的累加值，输出形状与输入形状完全相同。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`        | `Tensor`          | 参与计算的源张量，可为任意维度（N 维） |
| `dim`         | `int`             | 指定进行累积和操作的维度，必须在 `[-r, r-1]` 范围内（`r` 为输入维度数），负值表示从后往前索引 |
| `dtype`       | `ScalarType?`     | 可选，返回张量的期望数据类型。如果指定，则在操作前将输入转换为该 dtype，有助于防止溢出；默认为 `None`，此时整数类型默认提升为 `int64`，浮点类型保持不变 |
| 返回值      | `Tensor`| 一个新的张量，形状与输入张量完全一致，包含沿指定维度的累积和结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | √    | √    |  √     |  √    |  √    | √     | √      | √      | √     | √      | √      | √      |
| NPU    | √    | √    |  √     |  √    |  √    | √     | √      | √      | √     | √      | √      | √      |

> **说明**：除上表外，`float64`（`fp64`）及复数类型（`complex64`, `complex128`）同样支持。对于整数类型，未指定 `dtype` 时输出默认提升为 `int64`；对于布尔类型，在某些后端（如 MPS）内部会先转换为 `int32` 再计算。

#### 2.2.2 Shape 支持

1. **维度范围**：支持任意维度的张量（包括 0-D）。对于 0-D 张量，内部会先 `unsqueeze` 为 1-D 再执行操作。
2. **`dim` 参数**：必须为有效整数，且满足 `-r ≤ dim < r`（`r` 为输入维度数）。超出范围将抛出错误。
3. **空张量**：支持包含零元素的张量（即某个维度长度为 0），但需注意反向传播时需正确处理空张量情形。

### 2.3 特殊限制说明

> 1. 对于整数类型输入，若未指定 `dtype`，默认返回 `int64` 类型，这可能导致内存占用增加，建议根据实际需求显式指定 `dtype`。
> 2. `cumsum` 不是归约操作，因此输出形状与输入形状完全相同，不会减少任何维度。
> 3. 在某些后端（如 MPS）上，对于维度大于 4 的张量可能存在实现限制，建议在 GPU/CPU 上验证。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_cumsum_default(self, shape, dtype):
        def op_calc(tensor, dim):
            y = torch.ops.aten.cumsum.default(tensor, dim=dim)
            return y
        def op_calc_1(tensor, dim):
            y = torch.ops.aten.cumsum.default(tensor, dim=dim)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        # 选择有效维度，此处取 0（若 shape 为空，则需特殊处理，实际测试中可调整）
        dim = 0 if len(shape) > 0 else 0
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, dim)
```

**基础使用示例**：

```python
# 一维向量
a = torch.tensor([1, 2, 3, 4, 5])
y = torch.ops.aten.cumsum.default(a, dim=0)
# y = tensor([1, 3, 6, 10, 15])

# 二维张量，沿第 0 维（行方向）累积
a = torch.tensor([[1, 2], [3, 4], [5, 6]])
y = torch.ops.aten.cumsum.default(a, dim=0)
# y = tensor([[1, 2], [4, 6], [9, 12]])

# 指定 dtype 防止溢出
a = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
y = torch.ops.aten.cumsum.default(a, dim=0, dtype=torch.int64)
# y = tensor([1, 3, 6, 10], dtype=torch.int64)
```