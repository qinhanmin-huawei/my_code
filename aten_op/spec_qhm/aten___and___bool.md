# aten.__and__.bool

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.__and__.bool(bool self, bool other) -> bool
```

### 1.2 算子功能

1、`aten.__and__.bool` 对两个 bool 标量执行逻辑与（`a && b`），返回 bool 标量。

2、该 overload 的输入为 Python 标量，用例中借助辅助张量 `auxiliary` 参与构图。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `bool` | 第一个标量操作数。 |
| `other` | `bool` | 第二个标量操作数。 |
| `返回值` | `bool` | 逻辑与结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | x | x | x | √ |
| NPU | - | x | - | - | - | x | - | x | x | x | x | √ |

#### 2.2.2 Shape 支持

输入为标量；用例通过形状为 `(64,)` 的 bool 辅助张量触发编译。

### 2.3 特殊限制说明

仅支持 bool 标量输入。

### 2.4 使用方法

```python
    def test_aten___and___bool(self):
        def op_calc(auxiliary):
            y = torch.ops.aten.__and__.bool(True, True)
            return auxiliary | y

        def op_calc_1(auxiliary):
            y = torch.ops.aten.__and__.bool(True, True)
            y = auxiliary | y
            return torch.logical_or(y, y)

        auxiliary = torch.zeros((64,), dtype=torch.bool, device='npu')
        self.execute_inductor_test((64,), 'bool', op_calc, op_calc_1, auxiliary)
```