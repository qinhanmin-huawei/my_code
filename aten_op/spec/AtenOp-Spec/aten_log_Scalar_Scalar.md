# aten.log.Scalar_Scalar

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.log.Scalar_Scalar(Scalar a, Scalar b) -> float
```

### 1.2 算子功能

`aten.log.Scalar_Scalar` 计算以 `b` 为底的 `a` 的对数：`out = ln(a) / ln(b)`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `Scalar` | 真数 |
| `b` | `Scalar` | 底数 |
| 返回值 | `float` | 对数结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

下表按构造输入标量的来源 dtype 列示，实际参数为 Python 标量。

| 设备 | 参数 | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | `a`、`b` | √ | √ | √ | √ | √ | √ | × | × | × |
| NPU | `a`、`b` | √ | √ | √ | √ | √ | √ | × | × | × |

#### 2.2.2 Shape 支持

不适用，输入与返回值均为标量。

### 2.3 特殊限制说明

> 当前样例要求 `a > 0`、`b > 0` 且 `b != 1`。
>
> 当前测试中的标量计算被常量折叠，不作为设备端算子精度或性能验证。

### 2.4 使用方法

`_generate_scalar` 基于 `_generate_random_scalar` 生成正数 Python 标量，`avoid_one=True` 时避开 1。

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']

    @parametrize('dtype', dtypes)
    def test_aten_log_Scalar_Scalar(self, dtype):
        def op_calc(a, b, tensor):
            y = torch.ops.aten.log.Scalar_Scalar(a, b)
            return tensor + y

        def op_calc_1(a, b, tensor):
            y = torch.ops.aten.log.Scalar_Scalar(a, b)
            y = y + y
            return tensor + y

        a = self._generate_scalar(dtype)
        b = self._generate_scalar(dtype, avoid_one=True)
        tensor = torch.zeros(self._scalar_anchor_shape, dtype=getattr(torch, dtype), device='npu')
        self.execute_inductor_test(self._scalar_anchor_shape, dtype, op_calc, op_calc_1, a, b, tensor)
```
