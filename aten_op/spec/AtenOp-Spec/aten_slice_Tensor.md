# aten.slice.Tensor

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.slice.Tensor(
    Tensor(a) self,
    int dim=0,
    SymInt? start=None,
    SymInt? end=None,
    SymInt step=1
) -> Tensor(a)
```

### 1.2 算子功能

`aten.slice.Tensor` 沿 `dim` 截取张量，返回与 `self` 共享存储的切片视图：`self[..., start:end:step, ...]`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor(a)` | 输入张量 |
| `dim` | `int` | 切片维度，默认 0，范围为 [-self.ndim, self.ndim - 1] |
| `start` | `SymInt?` | 起始位置，包含；None 表示从头开始 |
| `end` | `SymInt?` | 结束位置，不包含；None 表示到该维末尾 |
| `step` | `SymInt` | 正整数步长，默认 1 |
| 返回值 | `Tensor(a)` | 与 self 共享存储，保持 dtype 和设备的视图 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、`self` 的维度范围：[1-4]。

2、输出维数不变，仅 `dim` 维长度由切片范围决定，其他维度与输入相同；空切片对应的维度长度为 0。

### 2.3 特殊限制说明

> `step > 0`；`start/end` 支持负索引和越界裁剪，不接受 0 维输入或非法 `dim`。返回值可能非连续，单独返回视图不一定生成设备 kernel，示例用 `y + y` 消费切片结果。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    slice_cases = [
        (0, None, None, 1),
        (0, 1, -1, 1),
        (-1, 1, -1, 2),
        (0, -8, None, 2),
        (0, -10000, 10000, 1),
    ]
    shape_slice_cases = []
    for shape in shapes:
        for slice_args in slice_cases:
            shape_slice_cases.append((shape, slice_args))
        for dim in range(1, len(shape) - 1):
            shape_slice_cases.append((shape, (dim, 1, -1, 2)))

    @parametrize('shape,slice_args', shape_slice_cases)
    @parametrize('dtype', dtypes)
    def test_aten_slice_Tensor(self, shape, dtype, slice_args):
        def op_calc(tensor, dim, start, end, step):
            y = torch.ops.aten.slice.Tensor(tensor, dim, start, end, step)
            return y

        def op_calc_1(tensor, dim, start, end, step):
            y = torch.ops.aten.slice.Tensor(tensor, dim, start, end, step)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        dim, start, end, step = slice_args
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, dim, start, end, step)
```
