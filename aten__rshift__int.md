# aten.__rshift__.int

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.__rshift__.int(int a, int b) -> int
```
### 1.2 算子功能
`aten.__rshift__.int` 是纯标量重载：对 int 标量 a 按位算术右移 b 位，返回 int。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `int` | 被移数（整数标量） |
| `b` | `int` | 位移量，须小于 64 |
| 返回值 | `int` | a >> b 的结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int32 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | √ | √ | √ | - | - | - | - | - | - |
| NPU | √ | √ | √ | - | - | - | - | - | - |

> 纯标量重载：参数固定为 int，返回 int，与 tensor dtype/shape 无关，不参与 tensor 的 dtype 判定。表中 dtype 为承载结果的载体 tensor 类型。

#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

> 纯标量重载在 inductor 下被常量折叠，不生成 kernel（`CAN NOT lowering` 为预期）。


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten___rshift___int(self, shape, dtype):
        def op_calc(first_element, scalar_a):
            y = torch.ops.aten.__rshift__.int(scalar_a, 2)
            result = first_element + y
            return result

        def op_calc_1(first_element, scalar_a):
            y = torch.ops.aten.__rshift__.int(scalar_a, 2)
            result = first_element + y + y
            return result

        first_element = torch.zeros(shape, dtype=getattr(torch, dtype), device="npu")
        scalar_a = self._generate_random_scalar(dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, scalar_a)
```
