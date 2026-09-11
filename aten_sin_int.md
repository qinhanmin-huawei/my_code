# aten.sin.int

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sin.int(int a) -> float
```

### 1.2 算子功能

`aten.sin.int` 计算整数标量 `a` 的正弦值：`out = sin(a)`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `int` | 输入标量，单位为弧度 |
| 返回值 | `float` | 正弦结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

下表按构造输入标量的来源 dtype 列示，实际参数为 Python 标量。

| 设备 | 参数 | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | `a` | √ | √ | √ | - | - | - | × | - | - |
| NPU | `a` | √ | √ | √ | - | - | - | × | - | - |

#### 2.2.2 Shape 支持

不适用，输入与返回值均为标量。

### 2.3 特殊限制说明

> `a` 须在 ATen int（有符号 64 位整数）范围内。
>
> 当前测试中的标量计算会发生常量折叠，不作为设备端算子精度或性能验证。

### 2.4 使用方法

`_generate_scalar` 基于 `_generate_random_scalar` 生成正数 Python 标量；`_scalar_anchor_shape = (128,)`。

```python
    dtypes = ['int8', 'int32', 'int64']

    @parametrize('dtype', dtypes)
    def test_aten_sin_int(self, dtype):
        def op_calc(a, tensor):
            y = torch.ops.aten.sin.int(a)
            return tensor + y

        def op_calc_1(a, tensor):
            y = torch.ops.aten.sin.int(a)
            y = y + y
            return tensor + y

        a = self._generate_scalar(dtype)
        tensor = torch.zeros(self._scalar_anchor_shape, dtype=torch.float32, device='npu')
        self.execute_inductor_test(self._scalar_anchor_shape, dtype, op_calc, op_calc_1, a, tensor)
```
