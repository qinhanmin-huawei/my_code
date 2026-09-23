# cond

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.higher_order.cond(pred, true_fn, false_fn, operands) -> Tensor
```

### 1.2 算子功能

根据 bool 条件执行一个分支。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `pred` | `Tensor` | 单元素 bool 条件。 |
| `true_fn / false_fn` | `Callable` | 条件分支。 |
| `operands` | `tuple[Tensor]` | 分支输入。 |
| `返回值` | `Tensor` | 所选分支结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | - | - | - | - | - | - | - | - | √ |
| NPU | - | - | - | - | - | - | - | - | - | - | - | √ |

#### 2.2.2 Shape 支持

输入 Tensor 支持 1～4 维。

### 2.3 特殊限制说明

GPU/NPU dtype 取决于分支中的算子，本例为加法与 clone。dtypes只用于构造计算的数据类型，不约束算子输入的condition。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['float32']
    predicates = [False, True]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    @parametrize('predicate', predicates)
    def test_cond(self, shape, dtype, predicate):
        def op_calc(condition, first_element):
            def true_fn(x):
                return x + x

            def false_fn(x):
                return x.clone()

            return torch.ops.higher_order.cond(condition, true_fn, false_fn, (first_element,))

        def op_calc_1(condition, first_element):
            def true_fn(x):
                return x + x

            def false_fn(x):
                return x.clone()

            y = torch.ops.higher_order.cond(condition, true_fn, false_fn, (first_element,))
            return y + y
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        condition = self._generate_tensor((), 'bool', min=-2, max=2)
        condition.fill_(predicate)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, condition, tensor1,
            op_name='torch.ops.higher_order.cond',
        )
```
