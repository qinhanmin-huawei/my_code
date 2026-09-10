# aten.relu

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.relu(Tensor self) -> Tensor
```

### 1.2 算子功能
对输入 Tensor 执行逐元素 ReLU 激活：当元素小于 0 时输出 0，否则保留原值。该操作返回新的 Tensor，不修改输入。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入 Tensor |
| 返回值 | `Tensor` | 与输入同 shape、同 dtype 的 ReLU 结果 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × | × | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × | - | - |

#### 2.2.2 Shape 支持
输入 `self` 支持 1-4 维，输出 shape 与输入一致。

### 2.3 特殊限制说明
> `aten.relu` 是 Tensor ReLU 调用形式；不支持 bool。GPU 不支持复数，NPU 复数类型不纳入本重载考察。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_relu(self, shape, dtype):
        def op_calc(tensor):
            return torch.ops.aten.relu(tensor)

        def op_calc_1(tensor):
            y = torch.ops.aten.relu(tensor)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor)
```