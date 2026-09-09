# aten.sign.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.sign.default(Tensor self) -> Tensor
```
### 1.2 算子功能
`aten.sign.default` 是 `aten.sign` 的默认重载（default overload），其功能与 `aten.sign` 一致：一个逐元素（Element-wise）数学算子，用于计算输入张量 `self` 中每个元素的符号：正数返回 1，负数返回 -1，零返回 0。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`        | `Tensor`          | 参与计算的源张量 |
| 返回值      | `Tensor`| 一个新的张量，形状与输入一致，包含每个元素的符号（1 / -1 / 0） |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | ×    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

> 注：aclnn不支持int8类型。
#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> NPU 不支持 int8 输入，GPU 支持 int8 输入。


### 2.4 使用方法


```python
    dtypes = ['int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_sign_default(self, shape, dtype):
        def op_calc(tensor, tensor2):
            y = torch.ops.aten.sign.default(tensor)
            return y

        def op_calc_1(tensor, tensor2):
            y = torch.ops.aten.sign.default(tensor)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        tensor2 = torch.full(shape, 2, dtype=getattr(torch, dtype), device="npu")
        self.optype = 'VV'
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, tensor2)
```
