# aten.floor

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.floor(Tensor self) -> Tensor
```
### 1.2 算子功能
1、`aten.floor` 是一个逐元素（Element-wise）的向下取整数学算子：对输入张量中的每个元素 \(x_i\)，计算 \(out_i = \lfloor x_i \rfloor\)，即返回**不大于 \(x_i\) 的最大整数**（向负无穷方向取整），例如 floor(2.3) = 2、floor(-2.3) = -3。
2、输出张量的形状与输入一致，结果与输入保持相同 dtype。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 参与计算的源张量 |
| 返回值 | `Tensor` | 一个新的张量，形状与输入一致，dtype 与输入相同，包含向下取整后的结果 |

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
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_floor(self, shape, dtype):
        def op_calc(first_element):
            y = torch.ops.aten.floor(first_element)
            return y

        def op_calc_1(first_element):
            y = torch.ops.aten.floor(first_element)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)
```
