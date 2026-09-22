# aten.select

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.select(Tensor(a) self, int dim, SymInt index) -> Tensor(a)
```

### 1.2 算子功能
沿 `dim` 维选择下标为 `index` 的单个切片并返回该切片的 view。被选择的维度从结果 shape 中移除，结果与输入共享存储，不修改输入。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入 Tensor |
| `dim` | `int` | 被选择的维度，支持负维度 |
| `index` | `SymInt` | 被选择的切片下标，支持合法负下标 |
| 返回值 | `Tensor` | 沿指定维度选择后的 view |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
输入支持 1-4 维 Tensor。若 `self.shape` 为 `(s0, ..., sn)`，选择维度 `dim=k` 后输出 shape 为 `(s0, ..., s{k-1}, s{k+1}, ..., sn)`。

### 2.3 特殊限制说明
> `dim` 必须是合法维度，`index` 必须落在对应维度范围内（允许负下标）。`select` 返回 view，通常不生成独立设备 kernel；本用例通过选择后的逐元素运算验证 view 的布局处理。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (16, 32), (4, 8, 16), (2, 4, 8, 16)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_select(self, shape, dtype):
        def op_calc(tensor, dim, index):
            return torch.ops.aten.select(tensor, dim, index)

        def op_calc_1(tensor, dim, index):
            y = torch.ops.aten.select(tensor, dim, index)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        dim = 0
        index = 0
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tensor, dim, index)
```