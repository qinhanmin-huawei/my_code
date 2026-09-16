# aten.broadcast_tensors.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.broadcast_tensors.default(Tensor[] tensors) -> Tensor[]
```
### 1.2 算子功能
`aten.broadcast_tensors.default` 是 `aten.broadcast_tensors` 的 default 重载，按广播语义将输入张量列表中的所有张量扩展为彼此兼容的统一形状，返回广播后的张量列表。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `tensors` | `Tensor[]` | 参与广播的张量列表 |
| 返回值 | `Tensor[]` | 广播后的张量列表，所有元素形状一致 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

### 2.3 特殊限制说明

> 输入张量需满足广播规则；返回值为张量列表，用例中取 `y[0]`、`y[1]` 参与计算。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]
    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_broadcast_tensors_default(self, shape, dtype):
        def op_calc(tensor, tensor2):
            y = torch.ops.aten.broadcast_tensors.default([tensor, tensor2])
            return y[0] + y[1]

        def op_calc_1(tensor, tensor2):
            y = torch.ops.aten.broadcast_tensors.default([tensor, tensor2])
            return y[0] + y[0] + y[1]

        tensor = self._generate_tensor(shape, dtype)
        tensor2 = torch.full(shape, 2, dtype=getattr(torch, dtype), device="npu")
        self.optype = 'VV'
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, tensor2)
```
