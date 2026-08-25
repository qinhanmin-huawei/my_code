# prims.copy_stride.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.prims.copy_stride.default(self: Tensor) -> Tensor
```
### 1.2 算子功能
`prims.copy_stride.default` 对输入张量执行逐元素拷贝，返回一个与原张量形状相同但存储布局连续（contiguous）的新张量。该操作是底层 prim 算子，确保输出张量具有标准的连续步长布局。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 源张量 |
| 返回值 | `Tensor` | 与原张量形状相同、数据相同但 stride 为连续布局的新张量 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

### 2.3 特殊限制说明

> 无

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_prims_copy_stride_default(self, shape, dtype):
        def op_calc(input):
            y = torch.ops.prims.copy_stride.default(input, input.stride())
            return y
        def op_calc_1(input):
            y = torch.ops.prims.copy_stride.default(input, input.stride())
            return y + y
        input = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input)
```