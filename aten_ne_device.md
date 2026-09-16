# aten.ne.device

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.ne.device(Device a, Device b) -> bool
```

### 1.2 算子功能
比较两个 `torch.device` 对象是否不相等，返回 Python `bool`。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `Device` | 第一个设备对象 |
| `b` | `Device` | 第二个设备对象 |
| 返回值 | `bool` | `a != b` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| 实现域 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Device object | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

该 overload 不接收 Tensor 或数值标量，因此没有适用的 dtype。
#### 2.2.2 Shape 支持
仅支持两个 `torch.device` 对象，不涉及 Tensor shape。

### 2.3 特殊限制说明
> 返回 Python `bool`。设备对象通常是编译期常量，比较可能被常量折叠；是否折叠以实际生成的图和 output code 为准。用例使用 `cpu` 与 `npu` 设备对象，并用 NPU Tensor 承接结果。

### 2.4 使用方法
```python
    def test_aten_ne_device(self):
        def op_calc(anchor, first, second):
            return anchor + torch.ops.aten.ne.device(first, second)

        def op_calc_1(anchor, first, second):
            result = torch.ops.aten.ne.device(first, second)
            return anchor + result + result

        anchor = torch.zeros((128,), dtype=torch.float32, device='npu')
        self.execute_inductor_test((128,), 'float32', op_calc, op_calc_1,
                                   anchor, torch.device('cpu'),
                                   torch.device('npu'), op_name='aten.add')
```