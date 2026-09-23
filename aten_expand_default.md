# aten.expand.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.expand.default(
    Tensor(a) self,
    SymInt[] size,
    *,
    bool implicit=False
) -> Tensor(a)
```

### 1.2 算子功能
按 `size` 对输入 Tensor 进行广播扩展并返回 view。大小为 1 的输入维度可以扩展到目标大小；扩展维度通过零步长共享输入存储，不复制数据。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor(a)` | 待扩展的输入 Tensor |
| `size` | `SymInt[]` | 目标 shape；可使用 `-1` 保留对应的输入维度大小 |
| `implicit` | `bool` | 是否为隐式广播，默认为 `False` |
| 返回值 | `Tensor(a)` | 与输入共享存储的扩展 view |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

#### 2.2.2 Shape 支持
输入支持 1-4 维 Tensor。`size` 的维度数必须不小于输入维度数；输入维度从右向左与目标维度对齐，各维大小必须相同或输入维度大小为 1。输出 shape 为 `size` 指定的 shape。

### 2.3 特殊限制说明
> `size` 中的 `-1` 只能保留已有输入维度，不能用于新增的前置维度。返回结果与输入共享存储，扩展维度的 stride 为 0，因此不应对扩展 view 进行可能产生写冲突的原地修改。该重载通常只产生视图元数据；用例通过其后的逐元素运算检查 Inductor 对扩展 view 的处理。用例采用默认 `implicit=False`。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(1,), (1, 32), (1, 8, 16), (1, 4, 8, 16)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_expand_default(self, shape, dtype):
        def op_calc(tensor, target_shape):
            return torch.ops.aten.expand.default(tensor, target_shape)

        def op_calc_1(tensor, target_shape):
            y = torch.ops.aten.expand.default(tensor, target_shape)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        target_shape = (4,) + shape[1:]
        self.execute_inductor_test(target_shape, dtype,
                                   op_calc, op_calc_1,
                                   tensor, target_shape)
```