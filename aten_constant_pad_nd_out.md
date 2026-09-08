# aten.constant_pad_nd.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.constant_pad_nd.out(Tensor self, int[] pad, Scalar value=0, *, Tensor(a!) out) -> Tensor(a!)
```
### 1.2 算子功能
`aten.constant_pad_nd.out` 是 `aten.constant_pad_nd` 的 out 变体（out variant），其功能与 `aten.constant_pad_nd` 一致：对任意维度的输入张量 `self` 进行常数填充（constant padding），并将计算结果直接覆盖写入到预先指定的 `out` 张量中。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`        | `Tensor`          | 参与计算的源张量，可为任意维度（N 维） |
| `pad`         | `int[]`           | 填充量列表，长度必须为偶数 `2 * K`，从最后一个维度开始，每两个数为一对 |
| `value`       | `Scalar`          | 填充的常数值，默认为 0 |
| `out`         | `Tensor`          | 输出张量，其数据类型（Dtype）和设备（Device）必须与输入张量匹配，形状必须与填充后的形状严格一致 |
| 返回值      | `Tensor`| `out` 张量本身 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`pad` 长度必须为偶数，且不超过输入维度的 2 倍；`out` 的形状必须与填充后的形状一致


### 2.3 特殊限制说明

> `pad` 为长度 `2 * K` 的列表，`out` 张量的形状必须与填充后的输出形状严格一致，数据类型和设备必须与输入匹配。


### 2.4 使用方法


```python
    def _get_out_shape(self, shape, pad):
        """计算填充后的形状"""
        out_shape = list(shape)
        for i in range(len(shape)):
            out_shape[i] += pad[2 * i] + pad[2 * i + 1]
        return out_shape

    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]
    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_constant_pad_nd_out(self, shape, dtype):
        def op_calc(first_element, pad, value, out):
            torch.ops.aten.constant_pad_nd.out(first_element, pad = pad, value = value, out=out)
            return out
        def op_calc_1(first_element, pad, value, out):
            torch.ops.aten.constant_pad_nd.out(first_element, pad = pad, value = value, out=out)
            return out + out
        first_element = self._generate_tensor(shape, dtype)
        if len(shape) == 1 :
            pad = [1, 1]
        elif len(shape) == 2 :
            pad = [1, 1, 2, 2]
        elif len(shape) == 3 :
            pad = [1, 1, 2, 2, 2, 2]
        elif len(shape) == 4:
            pad = [1, 1, 2, 2, 2, 2, 3, 3]
        out_shape = self._get_out_shape(shape, pad)
        out = self._generate_tensor(out_shape, dtype)
        value = 0
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, pad, value, out)
```
