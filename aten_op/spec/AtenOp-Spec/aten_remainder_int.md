# aten.remainder.float

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.remainder.float(self: Scalar, other: Scalar) -> Tensor
```


### 1.2 算子功能
`aten.remainder.float` 计算输入标量 `self` 与标量 `other` 的**逐元素取余（remainder）** 运算。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Scalar`（`float`） | 输入标量，支持浮点|
| `other` | `Scalar`（`int`） | 除数标量，支持浮点|
| 返回值 | `Scalar` | 取余的结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | ×      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | ×      |

#### 2.2.2 Shape 支持

输入只支持标量


### 2.3 特殊限制说明

> 无

