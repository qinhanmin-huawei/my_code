# aten._unsafe_index_put

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._unsafe_index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
```

### 1.2 算子功能

按第一维的整数索引写入 values，返回更新后的张量。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `indices` | `List[Optional[Tensor]]` | 索引张量或索引输出缓冲区。 |
| `values` | `Tensor` | 写入值。 |
| `accumulate` | `bool` | 算子配置参数，取值见使用方法。 |
| `返回值` | `Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

当accumulate为True时，不支持int8类型，因为会生成int8类型的原子加。与社区上游报错一致:atomic_add does not support int8。
#### 2.2.2 Shape 支持

1. 本测试覆盖的 shape 维度范围为 `[1, 4]`。
2. 测试 shape：`(4,)`、`(4, 8)`、`(4, 8, 16)`、`(4, 8, 16, 32)`。

### 2.3 特殊限制说明

索引只使用 int32/int64 且保证不越界；写入索引去重，避免重复写入的不确定性。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']
    index_dtypes = ['int32', 'int64']
    accumulates = [False, True]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    @parametrize('index_dtype', index_dtypes)
    @parametrize('accumulate', accumulates)
    def test_aten__unsafe_index_put(self, shape, dtype, index_dtype, accumulate):
        def op_calc(first_element, indices, values, accumulate):
            return torch.ops.aten._unsafe_index_put(first_element, [indices], values, accumulate)

        def op_calc_1(first_element, indices, values, accumulate):
            y = torch.ops.aten._unsafe_index_put(first_element, [indices], values, accumulate)
            return y + y
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        indices = self._generate_tensor((shape[0],), index_dtype, min=0, max=shape[0])
        indices = indices.sort().values.unique()
        values = self._generate_tensor((indices.numel(),) + tuple(shape[1:]), dtype, min=-2, max=2)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, indices, values, accumulate,
        )
```
