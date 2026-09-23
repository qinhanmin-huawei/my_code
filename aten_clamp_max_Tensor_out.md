# aten.clamp_max.Tensor_out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.clamp_max.Tensor_out(self: Tensor, max: Tensor, *, out: Tensor(a!)) -> Tensor(a!)
```

### 1.2 算子功能

逐元素计算 min(self, max)，将超过上界的元素截断到 max。结果写入 out。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 源张量。 |
| `max` | `Tensor` | 逐元素截断的上界。 |
| `out` | `Tensor` | 与输入 shape、dtype、设备一致的输出张量。 |
| `返回值` | `Tensor` | 与 out 共享存储的结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | x |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | x |

#### 2.2.2 Shape 支持

输入 Tensor 支持 1～4 维。

### 2.3 特殊限制说明

self 与 max 必须满足广播规则；本用例仅覆盖同 shape、同 dtype，不额外遍历广播形状。out 使用 empty_like(self) 创建，避免运行时调整输出形状。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(32,), (32, 32), (32, 32, 32), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_clamp_max_tensor_out(self, shape, dtype):
        def op_calc(first_element, second_element, out):
            torch.ops.aten.clamp_max.Tensor_out(first_element, second_element, out=out)
            return out

        def op_calc_1(first_element, second_element, out):
            torch.ops.aten.clamp_max.Tensor_out(first_element, second_element, out=out)
            return out + out

        tensor1 = self._generate_tensor(shape, dtype)
        tensor2 = self._generate_tensor(shape, dtype)
        out = torch.empty_like(tensor1)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, tensor2, out
        )
```
