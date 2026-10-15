# aten.repeat.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.repeat.default(Tensor self, SymInt[] repeats) -> Tensor
```

### 1.2 算子功能
沿各维重复复制输入 Tensor 的数据，返回新的 Tensor。该算子实际分配新存储，不返回与输入共享存储的 view。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 待重复复制的输入 Tensor |
| `repeats` | `SymInt[]` | 各维重复次数；每个值必须为非负整数 |
| 返回值 | `Tensor` | 按 `repeats` 复制后的新 Tensor |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

输出 dtype 与 `self.dtype` 相同；复数类型不纳入 NPU 本规格考察。

#### 2.2.2 Shape 支持
本规格覆盖 1-4 维输入 Tensor。`repeats` 长度必须不小于 `self.dim()`；长度不足时不满足 schema 的运行要求。若 `repeats` 更长，则在输入 shape 左侧补充长度为 1 的维度，输出各维为补齐后 shape 与 `repeats` 的逐维乘积。

### 2.3 特殊限制说明
> `repeats` 中的元素必须为非负整数。重复次数为 0 时，对应输出维度为 0；重复会产生新存储，不能按 view 算子处理。测试用例使用与输入维度数相同的 `repeats`，避免额外的左侧维度补齐分支。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (16, 32), (4, 8, 16), (2, 4, 8, 16)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_repeat_default(self, shape, dtype):
        def op_calc(tensor, repeats):
            return torch.ops.aten.repeat.default(tensor, repeats)

        def op_calc_1(tensor, repeats):
            y = torch.ops.aten.repeat.default(tensor, repeats)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        repeats = tuple(2 for _ in shape)
        
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tensor, repeats)
```