# aten.ne.enum

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.ne.enum(AnyEnumType a, AnyEnumType b) -> bool
```

### 1.2 算子功能
比较两个相同枚举类型的枚举值是否不相等，返回 Python `bool`。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `AnyEnumType` | 第一个枚举值 |
| `b` | `AnyEnumType` | 第二个枚举值，通常应与 `a` 属于同一枚举类型 |
| 返回值 | `bool` | `a != b` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| 实现域 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Enum object | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

该 overload 比较枚举对象，不涉及数值 dtype。
#### 2.2.2 Shape 支持
仅支持枚举值，不涉及 Tensor shape。

### 2.3 特殊限制说明
> `AnyEnumType` 是 ATen dispatcher 的内部泛型枚举类型，不等同于任意 Python `Enum`，也不能直接用 `torch._C._SDPBackend` 等 Python 枚举对象调用。当前 Python dispatcher 会在参数转换阶段报 `toIValue() cannot handle converting to type: AnyEnumType`，因此本重载无法通过现有 Python/Inductor 测试框架进行 eager、精度或性能验证。该限制发生在进入 Dynamo/Inductor 之前，不代表已经观察到 fallback 或 lowering 失败。

### 2.4 使用方法
当前 Python 测试框架没有可传入 `AnyEnumType` 的公开参数构造方式，不能提供可执行的 NPU 用例。直接使用下面的调用会在 eager 参数转换阶段失败：
```python
torch.ops.aten.ne.enum(
    torch._C._SDPBackend.MATH,
    torch._C._SDPBackend.EFFICIENT_ATTENTION,
)
# RuntimeError: toIValue() cannot handle converting to type: AnyEnumType
```

该 overload 如需验证，应在支持 ATen 内部枚举值的 C++ dispatcher/原生测试中构造参数；在当前 Python 测试目录中应标记为不可测试或跳过，不应使用 NPU Tensor `anchor` 掩盖参数转换失败。