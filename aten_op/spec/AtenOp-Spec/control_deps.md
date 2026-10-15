# control_deps

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.higher_order.control_deps(additional_deps, subgraph, *args, **kwargs) -> Tensor
```

### 1.2 算子功能

为子图建立额外执行顺序依赖，返回子图结果。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `additional_deps` | `tuple[Tensor]` | 必须先完成的张量依赖。 |
| `subgraph` | `GraphModule` | 需按依赖顺序执行的子图。 |
| `args` | `Tensor` | 子图入参。 |
| `返回值` | `Tensor` | 子图计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | x | x | x | x |
| NPU | - | x | - | - | - | x | - | x | x | x | x | x |

#### 2.2.2 Shape 支持

输入 Tensor 支持 1～4 维。

### 2.3 特殊限制说明

与社区报错一致：could not find kernel for HigherOrderOperator control_deps at dispatch key DispatchKey.AutogradCUDA/NPU。遗留问题：pytorch-2.10+L20报错,在高版本中支持。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_control_deps(self, shape, dtype):
        from torch._inductor.fx_passes.control_dependencies import control_deps

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.call_function(torch.ops.aten.add.Tensor, (x, x))
        graph.output(y)
        subgraph = torch.fx.GraphModule({}, graph)

        def op_calc(dependency, first_element):
            return torch.ops.higher_order.control_deps((dependency,), subgraph, first_element)

        def op_calc_1(dependency, first_element):
            y = torch.ops.higher_order.control_deps((dependency,), subgraph, first_element)
            return y + y
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        dependency = self._generate_tensor(shape, dtype, min=-2, max=2)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, dependency, tensor1,
            op_name='torch.ops.higher_order.control_deps',
        )
```
