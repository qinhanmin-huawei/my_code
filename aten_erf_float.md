# aten.erf.float

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.erf.float(float a) -> float
```
### 1.2 算子功能
`aten.erf.float` 是纯标量重载：对 float 标量 a 计算高斯误差函数 erf(a)，返回 float。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `float` | 输入标量 |
| 返回值 | `float` | erf(a) 的结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 参数 | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `a` | - | - | - | √ | √ | √ | - | - | - |


#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

无


### 2.4 使用方法


```python
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('dtype', dtypes)
    def test_aten_erf_float(self, shape, dtype):
        # aten::erf.float(float a) -> float
        def op_calc(first_element, scalar_a):
            y = torch.ops.aten.erf.float(scalar_a)
            result = first_element + y
            return result

        def op_calc_1(first_element, scalar_a):
            y = torch.ops.aten.erf.float(scalar_a)
            result = first_element + y + y
            return result

        first_element = torch.zeros(shape, dtype=getattr(torch, dtype), device="npu")
        scalar_a = self._generate_random_scalar(dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, scalar_a)
```
