# aten.__and__.int

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.__and__.int(int self, int other) -> int
```

### 1.2 算子功能

1、`aten.__and__.int` 对两个整型标量执行按位与，返回整型标量。

2、该 overload 的输入为 Python 标量，用例中借助辅助张量 `auxiliary` 参与构图。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `int` | 第一个标量操作数。 |
| `other` | `int` | 第二个标量操作数。 |
| `返回值` | `int` | 按位与结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | x | x | x | x |
| NPU | - | √ | - | - | - | √ | - | √ | x | x | x | x |

#### 2.2.2 Shape 支持

输入为标量；用例通过形状为 `(64,)` 的 int64 辅助张量触发编译。

### 2.3 特殊限制说明

仅支持整型标量输入，不支持浮点与 bool。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64']

    @parametrize('dtype', dtypes)
    def test_aten___and___int(self, dtype):
        def op_calc(auxiliary, first_element, other):
            return auxiliary + torch.ops.aten.__and__.int(first_element, other)

        def op_calc_1(auxiliary, first_element, other):
            y = auxiliary + torch.ops.aten.__and__.int(first_element, other)
            return y + y

        first_element = torch.tensor(6, dtype=getattr(torch, dtype)).item()
        other = torch.tensor(2, dtype=getattr(torch, dtype)).item()
        auxiliary = torch.zeros((64,), dtype=torch.int64, device='npu')
        self.execute_inductor_test((64,), dtype, op_calc, op_calc_1, auxiliary, first_element, other)
```