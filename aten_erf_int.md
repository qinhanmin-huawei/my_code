# aten.erf.int

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.erf.int(int a) -> float
```
### 1.2 算子功能
`aten.erf.int` 是纯标量重载：对 int 标量 a 计算高斯误差函数 erf(a)，返回 float（int 输入按浮点计算）。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `int` | 输入标量 |
| 返回值 | `float` | erf(a) 的结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 参数 | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `a` | √ | √ | √ | - | - | - | - | - | - |

#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

> 纯标量重载在 `torch.compile(inductor)` 下被常量折叠，不会生成 kernel。


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64']

    @parametrize('dtype', dtypes)
    def test_aten_erf_int(self, shape, dtype):
        def op_calc(first_element, scalar_a):
            y = torch.ops.aten.erf.int(scalar_a)
            result = first_element + y
            return result

        def op_calc_1(first_element, scalar_a):
            y = torch.ops.aten.erf.int(scalar_a)
            result = first_element + y + y
            return result

        first_element = torch.zeros(shape, dtype=getattr(torch, dtype), device="npu")
        scalar_a = self._generate_random_scalar(dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, scalar_a)
```
