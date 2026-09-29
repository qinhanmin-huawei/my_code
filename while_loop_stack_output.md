# while_loop_stack_output

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.higher_order.while_loop_stack_output(
    cond_fn: Callable, body_fn: Callable,
    carried_inputs: tuple, additional_inputs: tuple
) -> tuple[Tensor, ...]
```

此处记录纯 Tensor 状态的调用方式，通过 `torch._higher_order_ops.while_loop` 导入对应入口。

### 1.2 算子功能

循环更新状态，并将每次循环后的各项状态沿新首维分别堆叠：`y[k][t] = state_after_step_t[k]`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `cond_fn` | `Callable` | 接收当前状态与共享输入，返回标量 bool Tensor |
| `body_fn` | `Callable` | 返回下一轮状态 tuple |
| `carried_inputs` | `tuple[Tensor, ...]` | 初始状态；各项 shape 和 dtype 在迭代间保持一致 |
| `additional_inputs` | `tuple[Tensor, ...]` | 各轮共享的只读输入 |
| 返回值 | `tuple[Tensor, ...]` | 各状态项分别堆叠后的 Tensor |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | 数据状态 Tensor | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | 数据状态 Tensor | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| GPU | 计数器、循环上限 | - | - | - | - | - | - | - | √ | - | - | - | - |
| NPU | 计数器、循环上限 | - | - | - | - | - | - | - | √ | - | - | - | - |

数据 dtype 表对应示例加法循环体；其他循环体的支持类型由内部算子决定。

#### 2.2.2 Shape 支持

1、数据状态的支持维度范围：[1-4]；每项输出比对应状态多一个首维。

2、执行 `n > 0` 次时首维长度为 n；零次循环返回初始状态增加首维后的副本，首维长度为 1。

### 2.3 特殊限制说明

> 循环必须终止；当前规格使用不修改输入、不返回输入别名的循环体。输出首维长度由实际迭代次数决定。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    cases = [dict(shape=shape, iterations=3) for shape in shapes]
    cases.extend(
        dict(shape=shape, iterations=iterations)
        for shape in shapes[:1]
        for iterations in (0, 1)
    )

    @parametrize('case', cases)
    @parametrize('dtype', dtypes)
    def test_while_loop_stack_output(self, case, dtype):
        from torch._higher_order_ops.while_loop import while_loop_stack_output_op

        shape = case['shape']
        iterations = case['iterations']

        def cond_fn(index, tensor, limit):
            return index < limit

        def body_fn(index, tensor, limit):
            return index + 1, tensor + tensor

        def op_calc(index, tensor, limit):
            y = while_loop_stack_output_op(cond_fn, body_fn, (index, tensor), (limit,))
            return tuple(y)

        def op_calc_1(index, tensor, limit):
            y = while_loop_stack_output_op(cond_fn, body_fn, (index, tensor), (limit,))
            return tuple(part + part for part in y)

        if dtype == 'float16':
            tensor = self._generate_tensor(shape, dtype, min=-2000, max=2000)
        elif dtype == 'int8':
            tensor = self._generate_tensor(shape, dtype, min=-8, max=8)
        else:
            tensor = self._generate_tensor(shape, dtype)
        index = torch.zeros((), dtype=torch.int64, device=tensor.device)
        limit = torch.full((), iterations, dtype=torch.int64, device=tensor.device)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, index, tensor, limit)
```
