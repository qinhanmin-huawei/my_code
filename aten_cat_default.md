# aten.cat.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.cat.default(Tensor[] tensors, int dim=0) -> Tensor
```
### 1.2 算子功能
`aten.cat.default` 是 `aten.cat` 的 default 重载，沿指定维度 `dim` 拼接输入张量列表 `tensors`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `tensors` | `Tensor[]` | 参与拼接的张量列表 |
| `dim` | `int` | 拼接维度，默认 0 |
| 返回值 | `Tensor` | 拼接结果，`dim` 维大小为各输入之和 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

### 2.3 特殊限制说明

> 除 `dim` 维外，所有输入张量的形状必须一致。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]
    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_cat_default(self, shape, dtype):
        def op_calc(tensor, tensor2):
            y = torch.ops.aten.cat.default([tensor, tensor2], 0)
            return y + y
        def op_calc_1(tensor, tensor2):
            y = torch.ops.aten.cat.default([tensor, tensor2], 0)
            return y
        tensor = self._generate_tensor(shape, dtype)
        tensor2 = torch.full(shape, 2, dtype=getattr(torch, dtype), device="npu")
        self.optype = 'VV'
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, tensor2)
```
