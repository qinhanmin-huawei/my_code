# aten.copy.Dict_bool

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.copy.Dict_bool(
    Dict(bool, t)(a) self
) -> Dict(bool, t)
```

### 1.2 算子功能
返回以 bool 为键的 TorchScript 字典 `self` 的拷贝：创建一个新的字典对象，键值与原字典相同（浅拷贝，值对象本身共享），是字典层面的拷贝操作，不是设备 Tensor 计算。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Dict(bool, t)` | 待拷贝的字典，键为布尔标量，值为泛型类型 t |
| 返回值 | `Dict(bool, t)` | 新的字典对象，键值与 `self` 相同 |

### 2.2 支持规格
该重载依赖 TorchScript 字典类型 `Dict(bool, t)`，不定义 GPU/NPU Tensor dtype 和 shape 支持矩阵。

### 2.3 特殊限制说明
> 拷贝为浅拷贝，只复制字典容器本身，不复制键和值对象。该重载是 TorchScript 字典操作的历史接口，TorchScript 已被社区弃用，不纳入 Inductor 设备 Tensor 用例验证。

## 3. 遗留问题
该算子是 TorchScript 字典类型 `Dict(bool, t)` 的历史接口，TorchScript 架构已被社区弃用，无法构造用例。
