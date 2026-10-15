# aten.min.names_dim

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.min.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)
```

### 1.2 算子功能
按指定的 named tensor 维度求最小值和索引，返回 `values` 与 `indices` 二元组。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 带维度名称的输入 Tensor |
| `dim` | `str` | 单个归约维度的名称 |
| `keepdim` | `bool` | 是否保留该维度 |
| 返回值 | `(Tensor, Tensor)` | 最小值和索引 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | × | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

`indices` 输出 dtype 固定为 `int64`。

#### 2.2.2 Shape 支持
输出 shape 遵循指定命名维度的归约规则，`values` 和 `indices` shape 相同。

### 2.3 特殊限制说明
> 运行时 `dim` 类型为 `str`，输入必须携带匹配的 named Tensor 维度。当前 GPU/NPU TorchScript/eager 和 Inductor 对 named Tensor 支持有限，现有框架暂无法完成该重载的 lowering 验证。

### 2.4 使用方法
```python
    # 仅用于支持 named Tensor 的 eager 环境；当前 GPU/NPU 框架不作 Inductor 用例。
    x = torch.randn((4, 8), device='cpu').refine_names('N', 'C')
    values, indices = torch.ops.aten.min.names_dim(x, 'N', False)
```