# invoke_subgraph

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.higher_order.invoke_subgraph(
    subgraph: GraphModule,
    identifier: str | None,
    *operands: Tensor
) -> tuple[Tensor, ...]
```

此处记录纯张量输出的编译图入口；使用 `mark_compile_region` 标记函数，由编译器生成调用。

### 1.2 算子功能

执行已标记的子图并返回结果：`outputs = subgraph(*operands)`，用于重复计算区域的编译复用。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `subgraph` | `GraphModule` | 编译器捕获的计算子图 |
| `identifier` | `str / None` | 编译器生成的子图标识 |
| `operands` | `Tensor` | 子图的运行时输入 |
| 返回值 | `tuple[Tensor, ...]` | 子图的张量输出 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | operands | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | operands | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

上表为示例加法子图的输入规格；其他子图的 dtype 支持由其内部算子决定。

#### 2.2.2 Shape 支持

1、输入张量的支持维度范围：[1-4]。

2、输出 shape 由子图决定；示例保持输入 shape。

### 2.3 特殊限制说明

> 依赖提供 `mark_compile_region` 的 PyTorch 环境。当前规格使用无输入修改、无输入输出别名的纯函数子图。

### 2.4 使用方法

`mark_compile_region` 在 Eager 下直接执行函数，编译时生成 `invoke_subgraph`；示例连续调用同一子图两次。

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_invoke_subgraph(self, shape, dtype):
        from torch._higher_order_ops.invoke_subgraph import mark_compile_region

        @mark_compile_region
        def subgraph(tensor):
            return tensor + tensor

        def op_calc(tensor):
            y = subgraph(tensor)
            y = subgraph(y)
            return y

        def op_calc_1(tensor):
            y = subgraph(tensor)
            y = subgraph(y)
            return y + y

        if dtype == 'float16':
            tensor = self._generate_tensor(shape, dtype, min=-4000, max=4000)
        else:
            tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor)
```
