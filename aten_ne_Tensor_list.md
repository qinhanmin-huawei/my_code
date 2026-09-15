# aten.ne.Tensor_list

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.ne.Tensor_list(Tensor[] a, Tensor[] b) -> bool
```

### 1.2 算子功能
比较两个 Tensor 列表是否不相等，返回单个 Python `bool`。列表元素应为 Tensor，并按列表顺序和 Tensor 内容进行比较。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `Tensor[]` | 第一个 Tensor 列表 |
| `b` | `Tensor[]` | 第二个 Tensor 列表 |
| 返回值 | `bool` | 两个列表不相等时为 `True` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

返回值是 Python `bool`，不是 bool Tensor。

#### 2.2.2 Shape 支持
列表长度可不同；对应位 Tensor 应具有可比较的 shape、dtype 和 device。

### 2.3 特殊限制说明
> 该重载返回 Python `bool`。当列表长度或列表结构在编译时可知，且这些信息已足以确定比较结果时，该比较具备常量折叠条件。本用例传入长度分别为 1 和 2 的列表，结果可能在编译阶段确定；是否实际折叠以测试生成的代码为准。用例通过零值 Tensor 承接列表比较结果。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_ne_Tensor_list(self, shape, dtype):
        def op_calc(anchor, first, second):
            result = torch.ops.aten.ne.Tensor_list([first], [first, second])
            return anchor + result

        def op_calc_1(anchor, first, second):
            result = torch.ops.aten.ne.Tensor_list([first], [first, second])
            return anchor + result + result

        anchor = torch.zeros(shape, dtype=torch.float32, device='npu')
        first = self._generate_tensor(shape, dtype)
        second = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   anchor, first, second, op_name='aten.add')
```