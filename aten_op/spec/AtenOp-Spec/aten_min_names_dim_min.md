# aten.min.names_dim_min

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.min.names_dim_min(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
```

### 1.2 算子功能
按命名维度求最小值和索引，并写入 `min` 与 `min_indices` 输出。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 带维度名称的输入 Tensor |
| `dim` | `str` | 归约维度名称 |
| `keepdim` | `bool` | 是否保留归约维度 |
| `min` | `Tensor(a!)` | 最小值输出 Tensor |
| `min_indices` | `Tensor(b!)` | 索引输出 Tensor，dtype 为 `int64` |
| 返回值 | `(Tensor, Tensor)` | 返回值与两个 out Tensor 对应 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | × | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

`min_indices` 输出 dtype 必须为 `int64`，`min` 与输入数值 dtype 一致。

#### 2.2.2 Shape 支持
`min` 和 `min_indices` 的 shape 应与按命名维度归约后的 shape 一致。

### 2.3 特殊限制说明
> 当前 GPU/NPU TorchScript/eager 和 Inductor 对 named Tensor 支持有限，现有框架暂无法完成该重载的 lowering 验证。

### 2.4 使用方法
```python
    # 支持 named Tensor 的 eager 环境中，按 schema 使用两个关键字 out。
    x = torch.randn((4, 8), device='cpu').refine_names('N', 'C')
    values, indices = torch.ops.aten.min.names_dim_min(
        x, 'N', False,
        min=torch.empty((8,)),
        min_indices=torch.empty((8,), dtype=torch.int64))
```