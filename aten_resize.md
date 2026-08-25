# aten.resize

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.resize(self: Tensor, size: int[]) -> Tensor
```
### 1.2 算子功能
`aten.resize` 的核心功能是强行调整输入张量`self`的逻辑形状（Shape）到指定的大小`size`。若新尺寸超出原张量底层物理存储（Storage）的承载容量，该算子会在底层触发就地扩容。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self/input`        | `Tensor`          | 需要调整形状的源张量，操作会原地修改该张量 |
| `size`       | `int[]`    | 目标形状，各维度的长度列表 |
| 返回值      | `Tensor`| 调整后的张量，与 `self` 为同一张量 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> 无

### 2.4 使用方法


```python
    from functools import reduce
    import operator
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_resize(self, shape, dtype):
        def op_calc(input, new_size):
            y = torch.ops.aten.resize(input, size = new_size)
            return y
        def op_calc_1(input, new_size):
            y = torch.ops.aten.resize(input, size = new_size)
            return y + y
        input = self._generate_tensor(shape, dtype)
        num = reduce(operator.mul, shape)//2
        new_size = (2, num)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input, new_size)
```
