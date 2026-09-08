# aten.floor.Scalar

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.floor.Scalar(Scalar a) -> Scalar
```
### 1.2 算子功能
`aten.floor.Scalar` 是纯标量重载：对标量 a 向下取整，返回不大于 a 的最大整数（向负无穷方向取整），int/float 输入均返回 int 型结果。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `Scalar` | 输入标量（int/float） |
| 返回值 | `Scalar` | ⌊a⌋ 的结果，int 型 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 参数 | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `a` | √ | √ | √ | √ | √ | √ | - | - | - |

> 纯标量重载：`a` 固定为 Scalar（int/float），返回 int（int 恒等、float 向下取整），与 tensor dtype/shape 无关，不参与 tensor 的 dtype 判定。

#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

> 纯标量重载在 `torch.compile(inductor)` 下被常量折叠，不会生成 kernel。


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_floor_Scalar(self, shape, dtype):
        def op_calc(first_element, scalar_a):
            y = torch.ops.aten.floor.Scalar(scalar_a)
            result = first_element + y
            return result

        def op_calc_1(first_element, scalar_a):
            y = torch.ops.aten.floor.Scalar(scalar_a)
            result = first_element + y + y
            return result

        first_element = torch.zeros(shape, dtype=getattr(torch, dtype), device="npu")
        scalar_a = self._generate_random_scalar(dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, scalar_a)
```
