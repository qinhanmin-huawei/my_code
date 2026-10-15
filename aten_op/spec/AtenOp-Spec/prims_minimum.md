# prims.minimum

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.prims.minimum(self: Tensor, other: Tensor) -> Tensor
```
### 1.2 算子功能
`prims.minimum` 计算两个输入张量之间的**逐元素最小值（element-wise minimum）** 。其核心行为为：

- 对 `self` 和 `other` 中每个对应位置的元素进行比较，返回较小值。
- 若任一元素为 `NaN`，则结果对应位置为 `NaN`（即 `NaN` 优先于任何数值）。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self` / `input` | `Tensor` | 第一个输入张量 |
| `other` | `Tensor` | 第二个输入张量，须与 `self` 可广播且类型兼容 |
| 返回值 | `Tensor` | 新的张量，形状为广播后的形状，包含逐元素最小值结果 |


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
    def test_prims_minimum(self, shape, dtype):
        def op_calc(first_element, second_element):
            y = torch.ops.prims.minimum(first_element, second_element)
            return y

        def op_calc_1(first_element, second_element):
            y = torch.ops.prims.minimum(first_element, second_element)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, second_element)

```
