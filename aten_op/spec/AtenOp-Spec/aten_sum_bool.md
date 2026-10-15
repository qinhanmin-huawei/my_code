# aten.sum.bool

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sum.bool(bool[] self) -> int
```

### 1.2 算子功能

统计列表中 True 的数量，返回 Python `int`；空列表返回 `0`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `bool[]` | bool 元素组成的列表 |
| 返回值 | `int` | True 的数量 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

输入元素为 Python `bool`，输出为 Python `int`，不适用 GPU/NPU Tensor dtype 表。

#### 2.2.2 Shape 支持

不涉及 Tensor shape；列表长度可变，支持空列表。

### 2.3 特殊限制说明

> 当前测试中的列表求和会发生常量折叠，不作为设备端算子精度或性能验证。

### 2.4 使用方法

示例需先执行 `from torch.testing._internal.common_utils import parametrize, subtest`；`_generate_sum_values` 使用框架标量生成方法构造 Python `bool` 列表，或返回指定列表。

```python
    _scalar_anchor_shape = (128,)

    length_cases = [
        subtest(dict(length=0), name='empty'),
        subtest(dict(length=1), name='single'),
        subtest(dict(length=4), name='short'),
        subtest(dict(length=16), name='long'),
    ]
    bool_cases = length_cases + [
        subtest(dict(values=[False, False, False, False]), name='all_false'),
        subtest(dict(values=[True, True, True, True]), name='all_true'),
        subtest(dict(values=[True, False, True, False]), name='mixed'),
    ]

    def _generate_sum_values(self, case):
        if 'values' in case:
            return list(case['values'])
        return [bool(self._generate_random_scalar('bool')) for _ in range(case['length'])]

    @parametrize('case', bool_cases)
    def test_aten_sum_bool(self, case):
        def op_calc(values, tensor):
            y = torch.ops.aten.sum.bool(values)
            return tensor + y

        def op_calc_1(values, tensor):
            y = torch.ops.aten.sum.bool(values)
            y = y + y
            return tensor + y

        values = self._generate_sum_values(case)
        tensor = self._generate_tensor(self._scalar_anchor_shape, 'int64').zero_()
        self.execute_inductor_test(self._scalar_anchor_shape, 'int64', op_calc, op_calc_1, values, tensor)
```
