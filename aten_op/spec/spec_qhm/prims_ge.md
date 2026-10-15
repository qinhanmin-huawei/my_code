# prims.ge

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.prims.ge(Tensor self, Tensor other) -> Tensor
```
### 1.2 算子功能
1、`prims.ge` 是 prim 命名空间下的逐元素“大于等于”（`>=`）比较操作。

2、`self` 与 `other` 按广播规则对齐，逐元素比较，返回同形状的 bool 张量。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 第一个输入张量 |
| `other` | `Tensor` | 第二个输入张量，与 `self` 可广播 |
| 返回值 | `Tensor` | bool 张量，形状为广播后的 shape |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

> 表中为输入 `self` / `other` 的 dtype；输出**恒为 `bool`**，与输入 dtype 无关。

#### 2.2.2 Shape 支持

1、`self` 和 `other` 需满足广播规则

2、shape 维度范围：[1-4]

### 2.3 特殊限制说明

> 输出恒为 `bool`，与输入 dtype 无关。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_prims_ge(self, shape, dtype):
        def op_calc(x, y):
            res = torch.ops.prims.ge(x, y)
            return res
        def op_calc_plus(x, y):
            res = torch.ops.prims.ge(x, y)
            return res + res
        x = self._generate_tensor(shape, dtype)
        y = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_plus, x, y)
```