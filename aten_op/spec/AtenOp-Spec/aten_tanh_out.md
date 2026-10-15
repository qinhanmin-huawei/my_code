# aten.tanh.out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.tanh.out(self: Tensor, *, out: Tensor(a!)) -> Tensor(a!)
```

### 1.2 算子功能

逐元素计算输入的双曲正切值。 结果写入 `out`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 源张量。 |
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

`out` 必须与输入位于同一设备，且 dtype 必须能接收结果。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'float32', 'bfloat16']
    shapes = [(32,), (32, 32), (32, 32, 32), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_tanh_out(self, shape, dtype):
        def op_calc(first_element, out):
            x = first_element
            torch.ops.aten.tanh.out(x, out=out)
            return out

        def op_calc_1(first_element, out):
            x = first_element
            torch.ops.aten.tanh.out(x, out=out)
            return out + out

        tensor1 = self._generate_tensor(shape, dtype)
        output_dtype = dtype if dtype in TestUtils._test_float_dtypes else 'float32'
        out = torch.empty(shape, dtype=eval('torch.' + output_dtype), device=torch.device("npu"))
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1, out)
```
