# aten.erf.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.erf.default(Tensor self) -> Tensor
```
### 1.2 算子功能
1、`aten.erf.default` 是逐元素（Element-wise）的高斯误差函数（Gauss error function）算子：对输入张量中的每个元素 x 计算 y = erf(x)。结果在 (-1, 1) 内，erf(0) = 0、erf(+inf) = 1、erf(-inf) = -1，且为奇函数。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 参与计算的源张量 |
| 返回值 | `Tensor` | 一个新的张量，形状与输入一致，dtype 与浮点输入相同 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | ×    |  -     |  -    |  -    | ×     | -      | √      | √     | √      | √      | √      |
>注：对于int8和int32，eager报错不支持。


#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_erf_default(self, shape, dtype):
        def op_calc(first_element):
            y = torch.ops.aten.erf.default(first_element)
            return y

        def op_calc_1(first_element):
            y = torch.ops.aten.erf.default(first_element)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)
```
