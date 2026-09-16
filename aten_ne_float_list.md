# aten.ne.float_list

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.ne.float_list(float[] a, float[] b) -> bool
```

### 1.2 算子功能
按 Python 列表相等规则比较两个浮点列表是否不相等，返回 Python `bool`。列表长度、顺序和对应元素都会影响结果。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `float[]` | 第一个浮点列表 |
| `b` | `float[]` | 第二个浮点列表 |
| 返回值 | `bool` | `a != b` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| 实现域 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Host Scalar List | - | - | - | - | - | - | - | - | √ | √ | √ | - | - | - |

列表元素是 Python `float`，不涉及设备 Tensor dtype。
#### 2.2.2 Shape 支持
列表是一维 Python 容器；元素个数可变，不涉及 Tensor shape。

### 2.3 特殊限制说明
> 返回 Python `bool`。列表结构和元素为编译期常量时具备常量折叠条件；是否折叠以实际生成的图和 output code 为准。NPU Tensor 仅用于承接结果，不能代表该列表比较本身生成 NPU kernel。

### 2.4 使用方法
```python
    def test_aten_ne_float_list(self):
        def op_calc(anchor, first, second):
            return anchor + torch.ops.aten.ne.float_list(first, second)

        def op_calc_1(anchor, first, second):
            result = torch.ops.aten.ne.float_list(first, second)
            return anchor + result + result

        anchor = torch.zeros((128,), dtype=torch.float32, device='npu')
        self.execute_inductor_test((128,), 'float32', op_calc, op_calc_1,
                                   anchor, [1.0, 2.0], [1.0, 3.0],
                                   op_name='aten.add')
```