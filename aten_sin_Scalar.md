# aten.sin.Scalar

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.sin.Scalar(Scalar a) -> Scalar
```
### 1.2 算子功能
`aten.sin.Scalar` 是 `aten.sin` 的标量重载（Scalar overload），用于计算标量 `a` 的正弦值，返回标量结果。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `a`           | `Scalar`          | 参与计算的标量，以弧度为单位 |
| 返回值      | `Scalar`| 标量 `a` 的正弦值 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |

#### 2.2.2 Shape 支持

标量算子，不支持张量输入。


### 2.3 特殊限制说明

> 标量算子，不支持张量输入。


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']

    @parametrize('dtype', dtypes)
    def test_aten_sin_Scalar(self, dtype):
        def op_calc():
            y = torch.ops.aten.sin.Scalar(0.5)
            t = torch.tensor(y, device='npu', dtype=getattr(torch, dtype))
            return t + t

        def op_calc_1():
            y = torch.ops.aten.sin.Scalar(0.5)
            t = torch.tensor(y, device='npu', dtype=getattr(torch, dtype))
            return t

        self.execute_inductor_test(None, dtype, op_calc, op_calc_1)
```
