# aten.reshape

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)
```

### 1.2 算子功能
在保持元素总数不变的前提下，将输入 Tensor 转换为 `shape` 指定的形状。若输入的存储布局允许，新结果为与 `self` 共享存储的 view；否则算子会先复制数据再返回目标形状的 Tensor。因此不应依赖 `reshape` 一定返回 view。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor(a)` | 待改变形状的输入 Tensor |
| `shape` | `SymInt[]` | 目标 shape；最多允许一个维度取 `-1` 以自动推导 |
| 返回值 | `Tensor(a)` | shape 为目标形状、元素顺序与输入一致的 Tensor |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ | - | - |

输出 dtype 与 `self.dtype` 相同；复数类型不纳入本规格考察。

#### 2.2.2 Shape 支持
本规格覆盖 1-4 维输入 Tensor 和 1-4 维输出 Tensor。目标 shape 的元素数量必须与输入元素数量相同；若包含一个 `-1`，该维度由其余维度和输入元素总数自动推导。

### 2.3 特殊限制说明
> `shape` 中最多允许一个 `-1`，其他维度必须为合法的非负整数，推导后的元素总数必须等于 `self.numel()`。连续输入通常可以返回 view；非连续输入可能触发数据复制。该算子本身可能只改变元数据而不单独生成设备 kernel，本用例通过 reshape 后的逐元素加法验证 Inductor 对结果布局的处理。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (16, 32), (4, 8, 16), (2, 4, 8, 16)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_reshape(self, shape, dtype):
        def op_calc(tensor, target_shape):
            return torch.ops.aten.reshape(tensor, target_shape)

        def op_calc_1(tensor, target_shape):
            y = torch.ops.aten.reshape(tensor, target_shape)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        target_shape = (tensor.numel() // 2, 2)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tensor, target_shape)
```