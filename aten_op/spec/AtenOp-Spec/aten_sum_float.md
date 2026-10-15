# aten.sum.float

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sum.float(float[] self) -> float
```

### 1.2 算子功能

对浮点数列表求和，返回 Python `float`；空列表返回 `0.0`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `float[]` | float 元素组成的列表 |
| 返回值 | `float` | 列表元素之和 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

输入元素为 Python `float`，输出为 Python `float`，不适用 GPU/NPU Tensor dtype 表。

#### 2.2.2 Shape 支持

不涉及 Tensor shape；列表长度可变，支持空列表。

### 2.3 特殊限制说明

> 当前测试中的列表求和会发生常量折叠，不作为设备端算子精度或性能验证。

### 2.4 使用方法

示例需先执行 `from torch.testing._internal.common_utils import parametrize, subtest`；`_generate_sum_values` 使用框架标量生成方法构造 Python `float` 列表，或返回指定列表。

```python
    _scalar_anchor_shape = (128,)

    length_cases = [
        subtest(dict(length=0), name='empty'),
        subtest(dict(length=1), name='single'),
        subtest(dict(length=4), name='short'),
        subtest(dict(length=16), name='long'),
    ]
    float_cases = length_cases + [
        subtest(dict(values=[1.5, -2.0, 3.25]), name='mixed'),
        subtest(dict(values=[1.5, -1.5]), name='cancel'),
    ]

    def _generate_sum_values(self, case):
        if 'values' in case:
            return list(case['values'])
        return [float(self._generate_random_scalar('float32')) for _ in range(case['length'])]

    @parametrize('case', float_cases)
    def test_aten_sum_float(self, case):
        def op_calc(values, tensor):
            y = torch.ops.aten.sum.float(values)
            return tensor + y

        def op_calc_1(values, tensor):
            y = torch.ops.aten.sum.float(values)
            y = y + y
            return tensor + y

        values = self._generate_sum_values(case)
        tensor = self._generate_tensor(self._scalar_anchor_shape, 'float32').zero_()
        self.execute_inductor_test(self._scalar_anchor_shape, 'float32', op_calc, op_calc_1, values, tensor)
```
