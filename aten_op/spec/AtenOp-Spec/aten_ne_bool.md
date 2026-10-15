# aten.ne.bool

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.ne.bool(bool a, bool b) -> bool
```

### 1.2 算子功能
比较两个 Python 布尔值是否不相等，返回 Python `bool`。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `bool` | 第一个布尔参数 |
| `b` | `bool` | 第二个布尔参数 |
| 返回值 | `bool` | `a != b` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| 实现域 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Host Scalar | - | - | - | - | - | - | - | - | - | - | - | √ | - | - |

仅涉及 Python `bool`，不涉及 GPU/NPU Tensor dtype。

#### 2.2.2 Shape 支持
仅支持零维 Python 布尔值，无 Tensor shape。

### 2.3 特殊限制说明
> 返回 Python `bool`。当 `a` 和 `b` 在编译时是布尔常量时，该比较具备常量折叠条件，output code 可能不保留 `aten.ne.bool`；是否实际折叠以测试生成的代码为准。用例使用零值 Tensor 承接结果。

### 2.4 使用方法
```python
    def test_aten_ne_bool(self):
        def op_calc(anchor, first, second):
            return anchor + torch.ops.aten.ne.bool(first, second)

        def op_calc_1(anchor, first, second):
            y = torch.ops.aten.ne.bool(first, second)
            return anchor + y + y

        anchor = torch.zeros((128,), dtype=torch.float32, device='npu')
        self.execute_inductor_test((128,), 'float32', op_calc, op_calc_1,
                                   anchor, True, False, op_name='aten.add')
```