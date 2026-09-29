# prims.inductor_lookup_seed.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.prims.inductor_lookup_seed.default(Tensor seeds, int index) -> Tensor
```

### 1.2 算子功能

从种子向量中取出一个种子：`y = seeds[index]`，返回独立的零维 Tensor。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `seeds` | `Tensor` | 连续的一维种子张量 |
| `index` | `int` | 种子位置 |
| 返回值 | `Tensor` | 所选种子的零维副本 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | seeds | - | - | - | - | - | - | - | √ | - | - | - | - |
| NPU | seeds | - | - | - | - | - | - | - | √ | - | - | - | - |

种子采用 `inductor_seeds` 的 int64 格式，读取时保持整数值，不转换为浮点数。

#### 2.2.2 Shape 支持

`seeds` 的形状为 `(count,)`，`count > 0`；输出形状为 `()`。

### 2.3 特殊限制说明

> 当前规格使用 `0 <= index < count`。

### 2.4 使用方法

```python
    dtypes = ['int64']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    counts = list(dict.fromkeys(shape[-1] for shape in shapes))
    cases = [dict(count=count, index=0) for count in counts]
    if counts:
        cases.extend([
            dict(count=counts[0], index=counts[0] // 2),
            dict(count=counts[0], index=counts[0] - 1),
        ])

    @parametrize('case', cases)
    @parametrize('dtype', dtypes)
    def test_prims_inductor_lookup_seed_default(self, case, dtype):
        import torch._inductor.inductor_prims

        shape = (case['count'],)
        index = case['index']

        def op_calc(seeds, index):
            y = torch.ops.prims.inductor_lookup_seed.default(seeds, index)
            return y

        def op_calc_1(seeds, index):
            y = torch.ops.prims.inductor_lookup_seed.default(seeds, index)
            return y + y

        seeds = self._generate_tensor(shape, dtype, min=0, max=64)
        if seeds.dtype == torch.int64:
            seeds = seeds * (2**40) + 1
            seeds[0] = 0
            seeds[-1] = 2**40 + 1
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, seeds, index)
```
