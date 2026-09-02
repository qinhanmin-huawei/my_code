# aten.log2

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.log2(Tensor self) -> Tensor
```
### 1.2 算子功能
1、`aten.log2` 是一个用于计算以 2 为底的对数函数（\(y = log_2 x\)）的逐元素（Element-wise）数学算子，定义域为 \(x > 0\)。
2、输入为正数时返回正常对数值；输入为 +inf 时返回 inf；输入为 0 时返回 -inf；输入为负数或 NaN 时返回 NaN。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self/input`        | `Tensor`          | 参与计算的源张量 |
| `out`(返回值)      | `Tensor`| 一个新的张量，形状与输入一致，包含计算后的结果 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

无


### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_log2(self, shape, dtype):
        def op_calc(first_element):
            x = first_element
            y = torch.ops.aten.log2(x)
            result = y
            return result
        def op_calc_1(first_element):
            x = first_element + 1
            y = torch.ops.aten.log2(x)
            result = y + 1
            return result

        first_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)
```
