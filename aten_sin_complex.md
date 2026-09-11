# aten.sin.complex

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sin.complex(complex a) -> complex
```

### 1.2 算子功能

`aten.sin.complex` 计算复数标量 `a` 的正弦值：`out = sin(a)`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `complex` | 输入复数标量，单位为弧度 |
| 返回值 | `complex` | 复数正弦结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

下表按构造输入标量的来源 dtype 列示，实际参数为 Python 标量。

| 设备 | 参数 | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | `a` | - | - | - | - | - | - | × | √ | √ |
| NPU | `a` | - | - | - | - | - | - | × | √ | √ |

#### 2.2.2 Shape 支持

不适用，输入与返回值均为标量。

### 2.3 特殊限制说明

> 固定标量入图后会发生常量折叠，不作为设备端算子精度或性能验证。
>
> 本地 PyTorch 2.13.0 的 `fullgraph=True` 因 Python complex 返回值报错，默认模式发生 graph break；目标 NPU 环境待验证。

### 2.4 使用方法

`_generate_scalar` 用 `_generate_random_scalar` 生成实部，按 NumPy 来源 dtype 构造复数后返回 Python complex；`_scalar_anchor_shape = (128,)`。

```python
    dtypes = ['complex64', 'complex128']

    @parametrize('dtype', dtypes)
    def test_aten_sin_complex(self, dtype):
        def op_calc(a, tensor):
            y = torch.ops.aten.sin.complex(a)
            return tensor + y.real, tensor + y.imag

        def op_calc_1(a, tensor):
            y = torch.ops.aten.sin.complex(a)
            y = y + y
            return tensor + y.real, tensor + y.imag

        a = self._generate_scalar(dtype)
        tensor = torch.zeros(self._scalar_anchor_shape, dtype=torch.float32, device='npu')
        self.execute_inductor_test(self._scalar_anchor_shape, dtype, op_calc, op_calc_1, a, tensor)
```
