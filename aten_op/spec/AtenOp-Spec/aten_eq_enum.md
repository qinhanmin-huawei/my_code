# aten.eq.enum

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.eq.enum(AnyEnumType a, AnyEnumType b) -> bool
```
### 1.2 算子功能
1、`aten.eq.enum` 是纯标量重载：比较两个枚举值是否相等，返回 Python `bool`。

2、参数类型只认 TorchScript 枚举，Python 侧不可达。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `AnyEnumType` | 第一个枚举值 |
| `b` | `AnyEnumType` | 第二个枚举值 |
| 返回值 | `bool` | 比较结果（Python `bool`） |


### 2.2 支持规格

#### 2.2.1 DataType 支持

入参是 TorchScript 枚举（`AnyEnumType`），不是 Tensor，不适用 GPU/NPU Tensor dtype 表。

#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

1、 Python 侧不可达：`AnyEnumType` 只认 TorchScript 产出的 enum IValue，`enum.Enum` / `enum.IntEnum` / 裸 `int` / `str` 一律被拒。

2、 连 `torch.jit.script(Color)` 造出的 ScriptEnum 实参也被同样拒绝；唯一可用路径是 TorchScript 图内调用。

3、 在“Python 直接调用 + `torch.compile(backend='inductor')`”的用例框架下没有能通过的写法，实测仓未建测试方法，本文件不提供用例。


### 2.4 使用方法


```python
    # 实参只能由 TorchScript 图内产生，Python 侧无法构造，故不提供用例。
```
