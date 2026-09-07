# aten.add.Tensor

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
```
### 1.2 算子功能
`aten.add.Tensor` 是 `aten.add` 的张量重载（Tensor overload），其功能与 `aten.add` 一致：一个逐元素（Element-wise）数学算子，用于将张量 `self` 与张量 `other` 逐元素相加（`other` 先按 `alpha` 缩放，即 `self + alpha * other`）。`self` 与 `other` 需满足广播（broadcast）规则。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`        | `Tensor`          | 参与计算的第一个源张量 |
| `other`       | `Tensor`          | 参与计算的第二个源张量，与 `self` 需满足广播规则 |
| `alpha`       | `Scalar`          | `other` 的缩放系数，默认为 1 |
| 返回值      | `Tensor`| 一个新的张量，形状为 `self` 与 `other` 广播后的形状，包含相加后的结果 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`self` 与 `other` 的形状需满足广播规则（从后向前逐维比较，维度大小相等或其中一个为 1）


### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64','float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_add_Tensor(self, shape, dtype):
        def op_calc(first_element, second_element):
            y = torch.ops.aten.add.Tensor(first_element, second_element)
            return y
        def op_calc_1(first_element, second_element):
            y = torch.ops.aten.add.Tensor(first_element, second_element)
            return y + y
        tensor1 = self._generate_tensor(shape, dtype)
        tensor2 = torch.full(shape, 2, dtype=getattr(torch, dtype), device="npu")
        self.optype = 'VV'
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1, tensor2)
```
