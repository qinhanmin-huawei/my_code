# prims.sqrt

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.prims.sqrt(self: Tensor) -> Tensor
```
### 1.2 算子功能
`prims.sqrt`用于对输入张量（Tensor）执行逐元素（Element-wise）的平方根计算。

## 2. OP 规格

### 2.1 参数说明

| 参数名            | 类型                | 说明                                                             |
| -------------     | ----------------- | -------------------------------------------------------------- |
| `self/input`  | `Tensor`          | 需要进行逐元素（Element-wise）平方根计算的输入张量|
| `out`(返回值)      | `Tensor`| 一个新的张量，形状与输入一致，包含计算后的结果 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | ×    |  -     |  -    |  -    | ×     | -      | ×      | √     | √      | √      | ×      |
| NPU    | -    | ×    |  -     |  -    |  -    | ×     | -      | ×      | √     | √      | √      | ×      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_prims_sqrt(self, shape, dtype):
        def op_calc(first_element):
            x = first_element 
            y = torch.ops.prims.sqrt(x)
            result = y 
            return result
        def op_calc_1(first_element):
            x = first_element 
            y = torch.ops.prims.sqrt(x)
            result = y 
            return result + result
        tensor1 = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1)
```
