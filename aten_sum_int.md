# aten.sum.int

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sum.int(int[] self) -> int
```

### 1.2 算子功能

对整数列表求和，返回 Python `int`；空列表返回 `0`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `int[]` | int 元素组成的列表 |
| 返回值 | `int` | 列表元素之和 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

输入元素为 Python `int`，输出为 Python `int`，不适用 GPU/NPU Tensor dtype 表。

#### 2.2.2 Shape 支持

不涉及 Tensor shape；列表长度可变，支持空列表。

### 2.3 特殊限制说明

> 当前测试中的列表求和会发生常量折叠，不作为设备端算子精度或性能验证。当前实现存在 32 位整数累加溢出风险，各步累加结果应保持在 int32 范围内。

### 2.4 使用方法

示例需先执行 `from torch.testing._internal.common_utils import parametrize, subtest`；`_generate_sum_values` 使用框架标量生成方法构造 Python `int` 列表，或返回指定列表。

```python
    _scalar_anchor_shape = (128,)

    length_cases = [
        subtest(dict(length=0), name='empty'),
        subtest(dict(length=1), name='single'),
        subtest(dict(length=4), name='short'),
        subtest(dict(length=16), name='long'),
    ]
    int_cases = length_cases + [
        subtest(dict(values=[2, -3, 5]), name='mixed'),
        subtest(dict(values=[7, -7]), name='cancel'),
        subtest(dict(values=[2**24 + 1, -(2**24)]), name='large_integer'),
    ]

    def _generate_sum_values(self, case):
        if 'values' in case:
            return list(case['values'])
        return [int(self._generate_random_scalar('int8')) for _ in range(case['length'])]

    @parametrize('case', int_cases)
    def test_aten_sum_int(self, case):
        def op_calc(values, tensor):
            y = torch.ops.aten.sum.int(values)
            return tensor + y

        def op_calc_1(values, tensor):
            y = torch.ops.aten.sum.int(values)
            y = y + y
            return tensor + y

        values = self._generate_sum_values(case)
        tensor = self._generate_tensor(self._scalar_anchor_shape, 'int64').zero_()
        self.execute_inductor_test(self._scalar_anchor_shape, 'int64', op_calc, op_calc_1, values, tensor)
```
