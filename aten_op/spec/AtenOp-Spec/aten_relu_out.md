# aten.relu.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.relu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能
对输入 Tensor 执行逐元素 ReLU 激活，并将结果写入调用者提供的 `out` Tensor。该操作修改 `out`，不修改输入 `self`。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入 Tensor |
| `out` | `Tensor(a!)` | 保存结果的输出 Tensor，会被修改 |
| 返回值 | `Tensor(a!)` | 返回 `out` Tensor |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × | × | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × | - | - |

#### 2.2.2 Shape 支持
输入 `self` 和 `out` 支持 1-4 维，`out.shape` 必须与 `self.shape` 一致。

### 2.3 特殊限制说明
> `out` 的设备、shape 和 dtype 应与计算结果兼容；不支持 bool。GPU 不支持复数，NPU 复数类型不纳入本重载考察。

### 2.4 使用方法
```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_relu_out(self, shape, dtype):
        def op_calc(tensor, out_tensor):
            return torch.ops.aten.relu.out(tensor, out=out_tensor)

        def op_calc_1(tensor, out_tensor):
            y = torch.ops.aten.relu.out(tensor, out=out_tensor)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        out_tensor = torch.empty_like(tensor)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tensor, out_tensor)
```