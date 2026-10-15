# aten.ne.int

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.ne.int(int a, int b) -> bool
```

### 1.2 算子功能
比较两个 Python 整数标量是否不相等，返回 Python `bool`。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `int` | 第一个整数标量 |
| `b` | `int` | 第二个整数标量 |
| 返回值 | `bool` | `a != b` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| 实现域 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Host Scalar | - | √ | - | - | - | √ | - | √ | - | - | - | - | - | - |

仅涉及 Python `int`；uint 和 int16 不纳入测试范围。
#### 2.2.2 Shape 支持
仅支持 Python 整数标量，不涉及 Tensor shape。

### 2.3 特殊限制说明
> 返回 Python `bool`。两个整数为编译期常量时可能常量折叠；是否实际折叠以测试生成的图和 output code 为准。NPU Tensor 只承接比较结果。

### 2.4 使用方法
```python
    def test_aten_ne_int(self):
        def op_calc(anchor, first, second):
            return anchor + torch.ops.aten.ne.int(first, second)

        def op_calc_1(anchor, first, second):
            result = torch.ops.aten.ne.int(first, second)
            return anchor + result + result

        anchor = torch.zeros((128,), dtype=torch.float32, device='npu')
        self.execute_inductor_test((128,), 'float32', op_calc, op_calc_1,
                                   anchor, 1, 2, op_name='aten.add')
```