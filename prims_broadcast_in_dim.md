# prims.broadcast_in_dim

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.prims.broadcast_in_dim(
    Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions
) -> Tensor(a)
```

### 1.2 算子功能

按指定维度映射扩展张量并返回视图；映射轴保留原索引，长度为 1 的轴及新增轴按广播规则取值。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `Tensor(a)` | 输入张量 |
| `shape` | `SymInt[]` | 目标形状 |
| `broadcast_dimensions` | `int[]` | 输入各轴在目标形状中的对应轴 |
| 返回值 | `Tensor(a)` | 与输入共享存储的广播视图，dtype 不变 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

下表不代表 Inductor 编译测试已通过，当前阻塞见 2.3。

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | a | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | a | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、输入和输出的目标维度范围：[1-4]。

2、映射长度等于输入维数；目标维数不小于输入维数。

### 2.3 特殊限制说明

> 映射轴须非负、严格递增且不越界；对应输入轴长度须为 1 或等于目标轴长度。广播产生的共享存储位置不应独立写入。

已知编译限制：历史失败样例命中 [PyTorch #157610](https://github.com/pytorch/pytorch/issues/157610)，`torch.compile(..., backend="inductor")` 在 AOTAutograd functionalization 阶段因输出别名标注（`Tensor(a)`）报错，尚未完成编译结果的正确性与性能验证。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    cases = [
        dict(shape=shape, input_shape=(1, *shape[1:]), dimensions=list(range(len(shape))))
        for shape in shapes
    ]
    for shape in shapes[:1]:
        cases.append(dict(shape=shape, input_shape=shape, dimensions=list(range(len(shape)))))
    for shape in [shape for shape in shapes if len(shape) == 2][:1]:
        cases.append(dict(shape=shape, input_shape=(shape[1],), dimensions=[1]))
    for shape in [shape for shape in shapes if len(shape) == 3][:1]:
        cases.append(dict(shape=shape, input_shape=(shape[0], shape[2]), dimensions=[0, 2]))

    @parametrize('case', cases)
    @parametrize('dtype', dtypes)
    def test_prims_broadcast_in_dim(self, case, dtype):
        shape = case['shape']
        input_shape = case['input_shape']
        dimensions = case['dimensions']

        def op_calc(tensor, shape, dimensions):
            y = torch.ops.prims.broadcast_in_dim(tensor, shape, dimensions)
            return y

        def op_calc_1(tensor, shape, dimensions):
            y = torch.ops.prims.broadcast_in_dim(tensor, shape, dimensions)
            return y + y

        tensor = self._generate_tensor(input_shape, dtype)
        self.execute_inductor_test(input_shape, dtype, op_calc, op_calc_1, tensor, shape, dimensions)
```
