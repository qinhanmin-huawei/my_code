# aten.where.self_out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.where.self_out(
    Tensor condition,
    Tensor self,
    Tensor other,
    *,
    Tensor(a!) out
) -> Tensor(a!)
```

### 1.2 算子功能

按条件逐元素选择并写入 out：`y[i] = self[i] if condition[i] else other[i]`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `condition` | `Tensor` | bool 条件张量，True 选 self，False 选 other |
| `self` | `Tensor` | True 分支的输入张量 |
| `other` | `Tensor` | False 分支的输入张量 |
| `out` | `Tensor(a!)` | 独立分配的写回张量，返回值与其别名 |
| 返回值 | `Tensor(a!)` | 写入结果后的 out |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | condition | - | × | - | - | - | × | - | × | × | × | × | √ |
| GPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| GPU | other | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| GPU | out | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | condition | - | × | - | - | - | × | - | × | × | × | × | √ |
| NPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | other | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | out | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

当前规格要求 out 与选择结果的 dtype、shape、device 一致。

#### 2.2.2 Shape 支持

1、输入张量的目标维度范围：[1-4]。

2、所有张量输入须可广播，输出 shape 为广播后的形状。

### 2.3 特殊限制说明

> condition 必须为 bool，张量输入位于同一设备。 当前规格使用与输入独立的 out，仅测无梯度前向。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    cases = [dict(shape=shape) for shape in shapes]
    cases.extend(
        dict(shape=shape, fill=value)
        for shape in shapes[:1]
        for value in (0, 1)
    )

    broadcast_shapes = [shape for shape in shapes if len(shape) == 2][:1]
    tensor_cases = cases + [
        dict(shape=shape, broadcast=True)
        for shape in broadcast_shapes
    ]

    @parametrize('case', tensor_cases)
    @parametrize('dtype', dtypes)
    def test_aten_where_self_out(self, case, dtype):
        shape = case['shape']
        broadcast = case.get('broadcast', False)
        condition_shape = (1, shape[-1]) if broadcast else shape
        value_shape = (shape[0], 1) if broadcast else shape

        def op_calc(condition, a, b, out):
            y = torch.ops.aten.where.self_out(condition, a, b, out=out)
            return y

        def op_calc_1(condition, a, b, out):
            y = torch.ops.aten.where.self_out(condition, a, b, out=out)
            return y + y

        condition = self._generate_tensor(condition_shape, 'bool')
        if 'fill' in case:
            condition.fill_(case['fill'])
        a = self._generate_tensor(value_shape, dtype)
        b = self._generate_tensor(shape, dtype)
        out = torch.empty(shape, dtype=a.dtype, device=a.device)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, condition, a, b, out)
```
