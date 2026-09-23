# aten.where.Scalar

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.where.Scalar(Tensor condition, Scalar self, Scalar other) -> Tensor
```

### 1.2 算子功能

按条件逐元素选择输入值：`y[i] = self if condition[i] else other`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `condition` | `Tensor` | bool 条件张量，True 选 self，False 选 other |
| `self` | `Scalar` | True 分支的 Python 标量 |
| `other` | `Scalar` | False 分支的 Python 标量 |
| 返回值 | `Tensor` | 逐元素选择结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | condition | - | × | - | - | - | × | - | × | × | × | × | √ |
| GPU | self（Scalar） | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| GPU | other（Scalar） | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | condition | - | × | - | - | - | × | - | × | × | × | × | √ |
| NPU | self（Scalar） | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | other（Scalar） | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

Scalar 列按来源 dtype 填写，不表示 Python 标量保留该 dtype。同类标量输入时，bool 返回 bool，整数返回 int64，浮点返回默认浮点类型（通常为 fp32）。

#### 2.2.2 Shape 支持

1、输入张量的目标维度范围：[1-4]。

2、输出 shape 与 condition 一致。

### 2.3 特殊限制说明

> condition 必须为 bool；两个分支虽为标量，选择结果仍依赖运行时 Tensor，不属于纯标量常量折叠用例。

### 2.4 使用方法

`_generate_random_scalar(dtype)` 生成对应来源类型的 Python 标量，直接作为分支值传入。

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    cases = [dict(shape=shape) for shape in shapes]
    cases.extend(
        dict(shape=shape, fill=value)
        for shape in shapes[:1]
        for value in (0, 1)
    )

    @parametrize('case', cases)
    @parametrize('dtype', dtypes)
    def test_aten_where_Scalar(self, case, dtype):
        shape = case['shape']

        def op_calc(condition, a, b):
            y = torch.ops.aten.where.Scalar(condition, a, b)
            return y

        def op_calc_1(condition, a, b):
            y = torch.ops.aten.where.Scalar(condition, a, b)
            return y + y

        condition = self._generate_tensor(shape, 'bool')
        if 'fill' in case:
            condition.fill_(case['fill'])
        a = self._generate_random_scalar(dtype)
        b = self._generate_random_scalar(dtype)
        if a == b:
            b = type(b)(0 if a != 0 else 1)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, condition, a, b)
```
