# aten.log.float_complex

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.log.float_complex(float a, complex b) -> complex
```

### 1.2 算子功能

`aten.log.float_complex` 计算以复数 `b` 为底的浮点标量 `a` 的对数：`out = ln(a) / ln(b)`，复数对数取主值。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `float` | 真数 |
| `b` | `complex` | 底数 |
| 返回值 | `complex` | 复数对数结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

下表按构造输入标量的来源 dtype 列示，实际参数为 Python 标量。

| 设备 | 参数 | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | `a` | - | - | - | √ | √ | √ | × | - | - |
| GPU | `b` | - | - | - | - | - | - | × | √ | √ |
| NPU | `a` | - | - | - | √ | √ | √ | × | - | - |
| NPU | `b` | - | - | - | - | - | - | × | √ | √ |

#### 2.2.2 Shape 支持

不适用，输入与返回值均为标量。

### 2.3 特殊限制说明

> 当前样例要求 `a > 0`，并避开底数 `b = 0 + 0j` 与 `b = 1 + 0j`。
>
> 当前测试中的标量计算被常量折叠，不作为设备端算子精度或性能验证。

### 2.4 使用方法

`_generate_scalar` 基于 `_generate_random_scalar` 生成正数 Python 标量；`_scalar_anchor_shape = (128,)`。

```python
    def test_aten_log_float_complex(self):
        def op_calc(a, b, tensor):
            y = torch.ops.aten.log.float_complex(a, b)
            return tensor + y.real, tensor + y.imag

        def op_calc_1(a, b, tensor):
            y = torch.ops.aten.log.float_complex(a, b)
            y = y + y
            return tensor + y.real, tensor + y.imag

        a = self._generate_scalar('float32')
        b = complex(self._generate_scalar('float32'), 1.0)
        tensor = torch.zeros(self._scalar_anchor_shape, dtype=torch.float32, device='npu')
        self.execute_inductor_test(self._scalar_anchor_shape, 'float32', op_calc, op_calc_1, a, b, tensor)
```
