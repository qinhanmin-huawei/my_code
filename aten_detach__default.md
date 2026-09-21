# aten.detach_.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.detach_.default(Tensor(a!) self) -> Tensor(a!)
```
### 1.2 算子功能
1、`aten.detach_.default` 是 `aten.detach_` 的默认重载，将输入张量`self`本身从自动梯度计算图中分离（detached from computation graph），后续对该张量的操作不再追踪梯度。

2、不分配新张量，返回 `self` 本身。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor(a!)` | 源张量，会被原地修改（从计算图中分离） |
| 返回值 | `Tensor(a!)` | 返回 `self` 本身 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

### 2.3 特殊限制说明

> 无

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_detach__default(self, shape, dtype):
        def op_calc(tensor, tensor2):
            y = torch.ops.aten.detach_.default(tensor)
            return y + y

        def op_calc_1(tensor, tensor2):
            y = torch.ops.aten.detach_.default(tensor)
            return y

        tensor = self._generate_tensor(shape, dtype)
        tensor2 = torch.full(shape, 2, dtype=getattr(torch, dtype), device="npu")
        self.optype = 'VV'
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, tensor2)
```
