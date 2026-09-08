# aten.special_erf.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.special_erf.default(Tensor self) -> Tensor
```

### 1.2 算子功能

`aten.special_erf.default` 逐元素计算误差函数：`out[i] = erf(self[i])`，与 `aten.erf.default` 等价。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| 返回值 | `Tensor` | 与 self 同 shape 的结果张量 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |
| NPU | - | - | - | - | - | - | - | √ | √ | √ | √ | × |

> NPU 的 int8/int32 暂记为 `-`：当前环境下，Eager 路径调用的 `aclnnErf` 不支持这两种输入类型；Inductor 最小用例可执行。

#### 2.2.2 Shape 支持

1、shape 维度范围：[1-4]。

### 2.3 特殊限制说明

> 整数输入返回默认浮点类型（通常为 fp32），浮点输入保持 dtype；bool 不纳入当前支持规格。±Inf 分别返回 ±1，NaN 返回 NaN。

### 2.4 使用方法

`_generate_tensor(shape, dtype)` 用于生成对应设备的测试张量；当前 NPU 示例不包含 Eager 受限的 int8/int32。

```python
    dtypes = ['int64', 'float16', 'bfloat16', 'float32']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_special_erf_default(self, shape, dtype):
        def op_calc(tensor):
            y = torch.ops.aten.special_erf.default(tensor)
            return y

        def op_calc_1(tensor):
            y = torch.ops.aten.special_erf.default(tensor)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor)
```
