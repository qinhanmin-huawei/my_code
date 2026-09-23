# aten._unsafe_index.Tensor

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._unsafe_index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
```

### 1.2 算子功能

按第一维的整数索引读取元素。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `indices` | `List[Optional[Tensor]]` | 索引张量或索引输出缓冲区。 |
| `返回值` | `Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

输入 Tensor 支持 1～4 维。

### 2.3 特殊限制说明

索引只使用 int32/int64 且保证不越界；写入索引去重，避免重复写入的不确定性。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']
    index_dtypes = ['int32', 'int64']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    @parametrize('index_dtype', index_dtypes)
    def test_aten__unsafe_index_Tensor(self, shape, dtype, index_dtype):

        def op_calc(first_element, indices):
            return torch.ops.aten._unsafe_index.Tensor(first_element, [indices])

        def op_calc_1(first_element, indices):
            y = torch.ops.aten._unsafe_index.Tensor(first_element, [indices])
            return y + y
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        indices = self._generate_tensor((shape[0],), index_dtype, min=0, max=shape[0])
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, indices,
        )
```
