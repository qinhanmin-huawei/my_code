# aten.add.Scalar

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
```
### 1.2 算子功能
`aten.add.Scalar` 是 `aten.add` 的标量重载（Scalar overload），用于将张量 `self` 与标量 `other` 逐元素相加（`other` 先按 `alpha` 缩放，即 `self + alpha * other`）。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`        | `Tensor`          | 参与计算的源张量 |
| `other`       | `Scalar`          | 与 `self` 逐元素相加的标量 |
| `alpha`       | `Scalar`          | `other` 的缩放系数，默认为 1 |
| 返回值      | `Tensor`| 一个新的张量，形状与输入一致，包含相加后的结果 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64','float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_add_Scalar(self, shape, dtype):
        def op_calc(first_element):
            y = torch.ops.aten.add.Scalar(first_element, 3)
            return y
        def op_calc_1(first_element):
            y = torch.ops.aten.add.Scalar(first_element, 3)
            return y + y
        tensor1 = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor1)
```
