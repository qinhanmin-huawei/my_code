# aten.add.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
```
### 1.2 算子功能
`aten.add.out` 是 `aten.add.Tensor` 的 out 变体（out variant），其功能与 `aten.add.Tensor` 一致：将张量 `self` 与张量 `other` 逐元素相加（`other` 先按 `alpha` 缩放，即 `self + alpha * other`），并将计算结果直接覆盖写入到预先指定的 `out` 张量中。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`        | `Tensor`          | 参与计算的第一个源张量 |
| `other`       | `Tensor`          | 参与计算的第二个源张量，与 `self` 需满足广播规则 |
| `alpha`       | `Scalar`          | `other` 的缩放系数，默认为 1 |
| `out`         | `Tensor`          | 输出张量，其数据类型（Dtype）和设备（Device）必须与广播结果匹配，形状必须与广播后的形状一致 |
| 返回值      | `Tensor`| `out` 张量本身 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`self` 与 `other` 的形状需满足广播规则；`out` 的形状必须与广播后的形状一致


### 2.3 特殊限制说明

> `out` 张量的形状必须与广播后的输出形状严格一致，数据类型和设备必须与输入匹配。


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_add_out(self, shape, dtype):
        def op_calc(tensor, tensor2):
            out = torch.empty_like(tensor)
            torch.ops.aten.add.out(tensor, tensor2, out=out)
            return out + out

        def op_calc_1(tensor, tensor2):
            out = torch.empty_like(tensor)
            torch.ops.aten.add.out(tensor, tensor2, out=out)
            return out

        tensor = self._generate_tensor(shape, dtype)
        tensor2 = torch.full(shape, 2, dtype=getattr(torch, dtype), device="npu")
        self.optype = 'VV'
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, tensor2)
```
