# aten.select.int

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)
```

### 1.2 算子功能
沿整数维度 `dim` 选择下标为 `index` 的单个切片并返回 view；被选择维度从结果 shape 中移除。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入 Tensor |
| `dim` | `int` | 被选择的维度，支持负维度 |
| `index` | `SymInt` | 被选择的切片下标，支持合法负下标 |
| 返回值 | `Tensor` | 选择后的 view |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
输入支持 1-4 维；输出删除 `dim` 对应维度。

### 2.3 特殊限制说明
> `dim` 和 `index` 必须合法。该算子返回 view，不修改 `self`，非连续结果应按 view 处理。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (16, 32), (4, 8, 16), (2, 4, 8, 16)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_select_int(self, shape, dtype):
        def op_calc(tensor, dim, index):
            return torch.ops.aten.select.int(tensor, dim, index)

        def op_calc_1(tensor, dim, index):
            y = torch.ops.aten.select.int(tensor, dim, index)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tensor, 0, 0)
```