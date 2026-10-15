# aten.sub.out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sub.out(self: Tensor, other: Tensor, *, Scalar alpha=1, out: Tensor(a!)) -> Tensor(a!)
```

### 1.2 算子功能

逐元素计算 `self - alpha * other`，并将结果写入 `out`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 被减数张量。 |
| `other` | `Tensor` | 减数张量。 |
| `alpha` | `Scalar` | `other` 的缩放系数，默认值为 1。 |
| `out` | `Tensor` | 承接结果的可写张量。 |
| `返回值` | `Tensor` | 与 `out` 别名相同的张量。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |



#### 2.2.2 Shape 支持

输入 Tensor 支持 1～4 维。

### 2.3 特殊限制说明

bool Tensor 不支持减法；`out` 必须与输入位于同一设备且 dtype 能接收类型提升后的结果。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool']
    shapes = [(32,), (32, 32), (32, 32, 32), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_sub_out(self, shape, dtype):
        def op_calc(first_element, second_element, out):
            x = first_element
            torch.ops.aten.sub.out(x, second_element, alpha=2, out=out)
            return out

        def op_calc_1(first_element, second_element, out):
            x = first_element
            torch.ops.aten.sub.out(x, second_element, alpha=2, out=out)
            return out + out

        tensor1 = self._generate_tensor(shape, dtype)
        tensor2 = self._generate_tensor(shape, dtype)
        out = torch.empty_like(tensor1)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1, tensor2, out)
```
