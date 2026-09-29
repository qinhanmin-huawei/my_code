# foreach_map

## 1. OP 概述

### 1.1 算子原型

```text
foreach_map(op: Callable, *operands) -> list[Tensor]
```

通过 `torch._higher_order_ops.foreach_map` 导入；编译图对应 `torch.ops.higher_order.foreach_map`。

### 1.2 算子功能

对张量列表逐项应用函数：`y[i] = op(a[i], b[i], ...)`，共享参数用于每一项。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `op` | `Callable` | 对每组参数执行的逐元素函数 |
| `operands` | `list[Tensor] / Tensor / Scalar` | 张量列表按位置配对，非列表参数在各项之间共享 |
| 返回值 | `list[Tensor]` | 按列表顺序返回各项结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | operands 中的 Tensor | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | operands 中的 Tensor | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

上表为示例加法函数的输入规格；其他函数的 dtype 支持由其内部算子决定。

#### 2.2.2 Shape 支持

1、列表内张量的支持维度范围：[1-4]。

2、当前规格使用非空、等长列表；各组张量输入须可广播。示例列表长度为 2。

### 2.3 特殊限制说明

> 当前规格使用不修改输入、不返回输入别名的逐元素函数；同组 Tensor 位于同一设备。

### 2.4 使用方法

样例同时覆盖两个列表逐项配对，以及所有列表项共享一个 Tensor 参数。

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    cases = [dict(shape=shape) for shape in shapes]
    if shapes:
        cases.append(dict(shape=shapes[0], shared=True))

    @parametrize('case', cases)
    @parametrize('dtype', dtypes)
    def test_foreach_map(self, case, dtype):
        from torch._higher_order_ops.foreach_map import foreach_map

        shape = case['shape']
        shared = case.get('shared', False)

        def op_calc(a, b, c, shared):
            other = c if shared else [b, c]
            y = foreach_map(torch.add, [a, b], other)
            return tuple(y)

        def op_calc_1(a, b, c, shared):
            other = c if shared else [b, c]
            y = foreach_map(torch.add, [a, b], other)
            return tuple(part + part for part in y)

        a = self._generate_tensor(shape, dtype)
        b = self._generate_tensor(shape, dtype)
        c = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, a, b, c, shared)
```
