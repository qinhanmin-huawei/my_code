# aten._foreach_rsqrt

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._foreach_rsqrt(self: Tensor[]) -> Tensor[]
```

### 1.2 算子功能

计算输入列表中每个张量的逐元素平方根倒数。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor[]` | 输入张量列表。 |
| `返回值` | `Tensor[]` | 与输入列表等长的结果张量列表。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |


#### 2.2.2 Shape 支持

1. `shape` 遍历值：`(4,)`、`(4, 8)`、`(4, 8, 16)`、`(4, 8, 16, 32)`。
2. 输入 Tensor 列表长度固定为 `2`，每个元素使用相同的遍历 shape。输入值范围固定为 `[1, 8)`。

### 2.3 特殊限制说明

输入值固定为正数以保证平方根倒数有定义；bool 对该运算无意义。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__foreach_rsqrt(self, shape, dtype):
        def op_calc(tensors):
            return torch.ops.aten._foreach_rsqrt(tensors)

        def op_calc_1(tensors):
            result = torch.ops.aten._foreach_rsqrt(tensors)
            return [item + item for item in result]

        tensors = [self._generate_tensor(shape, dtype, min=1, max=8) for _ in range(2)]
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensors)
```
