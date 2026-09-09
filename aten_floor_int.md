# aten.floor.int

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.floor.int(int a) -> int
```
### 1.2 算子功能
`aten.floor.int` 是纯标量重载：对 int 标量 a 向下取整（整数恒等返回），返回 int。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `int` | 输入标量 |
| 返回值 | `int` | ⌊a⌋ 的结果（恒等返回） |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 参数 | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `a` | √ | √ | √ | - | - | - | - | - | - |

> 纯标量重载：`a` 固定为 int，返回 int（整数恒等返回），与 tensor dtype/shape 无关，不参与 tensor 的 dtype 判定。

#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

> 纯标量重载在 `torch.compile(inductor)` 下被常量折叠，不会生成 kernel。


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_floor_int(self, shape, dtype):
        def op_calc(first_element, scalar_a):
            y = torch.ops.aten.floor.int(scalar_a)
            result = first_element + y
            return result

        def op_calc_1(first_element, scalar_a):
            y = torch.ops.aten.floor.int(scalar_a)
            result = first_element + y + y
            return result

        first_element = torch.zeros(shape, dtype=getattr(torch, dtype), device="npu")
        scalar_a = self._generate_random_scalar(dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, scalar_a)
```
