# aten.div.Tensor

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.div.Tensor(self: Tensor, other: Tensor) -> Tensor
```
### 1.2 算子功能
1、`aten.div.Tensor` 是逐元素（Element-wise）的真除法（true division）算子：对输入张量 `self` 与 `other` 中对应元素执行除法 out = self / other，支持广播。
2、结果 dtype：int输入提升为 float32；浮点输入结果与输入同 dtype。
3、浮点除数为 0 时遵循 IEEE 语义：非零有限值除以 0 得 ±inf，0/0 得 nan。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 被除数张量 |
| `other` | `Tensor` | 除数张量，须与 `self` 可广播（broadcastable） |
| 返回值 | `Tensor` | 一个新的张量，形状为广播后的形状；int输入时 dtype 为 float32，浮点输入时与输入同 dtype |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |


#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`self` 与 `other` 支持广播，输出张量形状为广播后的形状


### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_div_Tensor(self, shape, dtype):
        def op_calc(first_element, second_element):
            y = torch.ops.aten.div.Tensor(first_element, second_element)
            return y

        def op_calc_1(first_element, second_element):
            y = torch.ops.aten.div.Tensor(first_element, second_element)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype, min=1) if dtype in _INT_DTYPES else self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, second_element)
```
