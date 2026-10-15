# prims.convert_element_type.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.prims.convert_element_type.default(Tensor a, ScalarType dtype) -> Tensor
```

### 1.2 算子功能

将张量复制并转换为指定 dtype：`y[i] = cast(a[i], dtype)`，shape 和 device 不变；dtype 相同时也返回独立副本。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `Tensor` | 输入张量 |
| `dtype` | `ScalarType` | 目标 dtype |
| 返回值 | `Tensor` | 转换后的张量 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | a | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | a | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| GPU | dtype（输出） | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | dtype（输出） | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

输入和输出的支持维度范围：[1-4]，输出 shape 与输入一致。

### 2.3 特殊限制说明

> 浮点转整数向零截断，输入须为有限值且位于目标整数范围内；转 bool 时零为 False、非零为 True。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    cases = [dict(shape=shape, target_dtype='float32') for shape in shapes]
    if shapes:
        largest_shape = max(shapes, key=lambda shape: torch.Size(shape).numel())
        for target_dtype in dtypes:
            if target_dtype != 'float32':
                cases.append(dict(shape=largest_shape, target_dtype=target_dtype))

    @parametrize('case', cases)
    @parametrize('dtype', dtypes)
    def test_prims_convert_element_type_default(self, case, dtype):
        shape = case['shape']
        target_dtype = getattr(torch, case['target_dtype'])

        def op_calc(tensor, target_dtype):
            y = torch.ops.prims.convert_element_type.default(tensor, target_dtype)
            return y

        def op_calc_1(tensor, target_dtype):
            y = torch.ops.prims.convert_element_type.default(tensor, target_dtype)
            return y + y

        if target_dtype == torch.int8:
            tensor = self._generate_tensor(shape, dtype, min=-64, max=63)
        else:
            tensor = self._generate_tensor(shape, dtype)
        flat_tensor = tensor.reshape(-1)
        flat_tensor[0] = 0
        if tensor.is_floating_point():
            flat_tensor[1:2] = -1.5
            flat_tensor[2:3] = 1.5
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, target_dtype)
```
