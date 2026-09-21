# aten._foreach_sign.out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._foreach_sign.out(self: Tensor[], *, out: Tensor(a!)[]) -> ()
```

### 1.2 算子功能

计算输入列表中每个张量元素的符号。 结果写入 `out`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor[]` | 输入张量列表。 |
| `out` | `Tensor[]` | 可写输出张量列表。 |
| `返回值` | `()` | 算子通过 `out` 写回结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |


#### 2.2.2 Shape 支持

1. `shape` 遍历值：`(4,)`、`(4, 8)`、`(4, 8, 16)`、`(4, 8, 16, 32)`。
2. 输入 Tensor 列表长度固定为 `2`，每个元素使用相同的遍历 shape。输入值范围固定为 `[-8, 8)`。 `out` 列表长度固定为 `2`，元素 shape 和 dtype 与对应输入相同。

### 2.3 特殊限制说明

不覆盖复数和 bool。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__foreach_sign_out(self, shape, dtype):
        def op_calc(tensors, out):
            torch.ops.aten._foreach_sign.out(tensors, out=out)
            return out

        def op_calc_1(tensors, out):
            torch.ops.aten._foreach_sign.out(tensors, out=out)
            return [item + item for item in out]

        tensors = [self._generate_tensor(shape, dtype, min=-8, max=8) for _ in range(2)]
        out = [torch.empty_like(item) for item in tensors]
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensors, out)
```
