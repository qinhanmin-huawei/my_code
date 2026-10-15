# aten.sum.complex

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sum.complex(complex[] self) -> complex
```

### 1.2 算子功能

对复数列表求和，返回 Python `complex`；空列表返回 `0j`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `complex[]` | complex 元素组成的列表 |
| 返回值 | `complex` | 列表元素之和 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

输入元素为 Python `complex`，输出为 Python `complex`，不适用 GPU/NPU Tensor dtype 表。

#### 2.2.2 Shape 支持

不涉及 Tensor shape；列表长度可变，支持空列表。

### 2.3 特殊限制说明

> 当前测试中的列表求和会发生常量折叠，不作为设备端算子精度或性能验证。

### 2.4 使用方法

示例需先执行 `from torch.testing._internal.common_utils import parametrize, subtest`；`_generate_sum_values` 使用框架标量生成方法构造 Python `complex` 列表，或返回指定列表。

```python
    _scalar_anchor_shape = (128,)

    length_cases = [
        subtest(dict(length=0), name='empty'),
        subtest(dict(length=1), name='single'),
        subtest(dict(length=4), name='short'),
        subtest(dict(length=16), name='long'),
    ]
    complex_cases = length_cases + [
        subtest(dict(values=[1 + 2j, 3 - 1j]), name='mixed'),
        subtest(dict(values=[1 + 2j, -1 - 2j]), name='cancel'),
    ]

    def _generate_sum_values(self, case):
        if 'values' in case:
            return list(case['values'])
        return [complex(self._generate_random_scalar('float32'), self._generate_random_scalar('float32')) for _ in range(case['length'])]

    @parametrize('case', complex_cases)
    def test_aten_sum_complex(self, case):
        def op_calc(values, tensor):
            y = torch.ops.aten.sum.complex(values)
            return tensor + y.real, tensor + y.imag

        def op_calc_1(values, tensor):
            y = torch.ops.aten.sum.complex(values)
            y = y + y
            return tensor + y.real, tensor + y.imag

        values = self._generate_sum_values(case)
        tensor = self._generate_tensor(self._scalar_anchor_shape, 'float32').zero_()
        self.execute_inductor_test(self._scalar_anchor_shape, 'float32', op_calc, op_calc_1, values, tensor)
```
