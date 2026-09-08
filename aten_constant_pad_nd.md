# aten.constant_pad_nd

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor
```
### 1.2 算子功能
`aten.constant_pad_nd` 的核心功能是对任意维度的输入张量 `self` 进行常数填充（constant padding）：按 `pad` 指定的各维度前后填充量，用常数值 `value` 在张量的末尾 `len(pad)/2` 个维度上进行填充，返回填充后的新张量。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`        | `Tensor`          | 参与计算的源张量，可为任意维度（N 维） |
| `pad`         | `int[]`           | 填充量列表，长度必须为偶数 `2 * K`，从最后一个维度开始，每两个数为一对  |
| `value`       | `Scalar`          | 填充的常数值，默认为 0 |
| 返回值      | `Tensor`| 一个新的张量，维度与输入一致，被填充的 `K` 个维度尺寸增加相应的填充量，其余维度尺寸不变 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`pad` 长度必须为偶数，且不超过输入维度的 2 倍（即最多填充全部维度）


### 2.3 特殊限制说明

> `pad` 为长度 `2 * K` 的列表。


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_constant_pad_nd(self, shape, dtype):
        def op_calc(tensor, pad, value):
            y = torch.ops.aten.constant_pad_nd(tensor, pad=pad, value=value)
            return y
        def op_calc_1(tensor, pad, value):
            y = torch.ops.aten.constant_pad_nd(tensor, pad=pad, value=value)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        pad = [1, 1] + ([2, 2] if len(shape) >= 2 else [])
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, pad, value=0)
```
