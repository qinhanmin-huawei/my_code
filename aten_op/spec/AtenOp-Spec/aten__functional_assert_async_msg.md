# aten._functional_assert_async.msg

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._functional_assert_async.msg(Tensor self, str assert_msg, Tensor dep_token) -> Tensor
```

### 1.2 算子功能

检查条件是否为真。返回依赖张量以表达执行依赖。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `assert_msg` | `str` | 断言消息。 |
| `dep_token` | `Tensor` | 依赖张量。 |
| `返回值` | `Tensor` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | x | x | x | x |
| NPU | - | x | - | - | - | x | - | x | x | x | x | x |

#### 2.2.2 Shape 支持

输入 Tensor 支持 1～4 维。

### 2.3 特殊限制说明

报错与社区上游一致：LoweringException: TypeError: lower_assert_functional_async() takes 2 positional arguments but 3 were given。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__functional_assert_async_msg(self, shape, dtype):

        def op_calc(condition, message, token):
            return torch.ops.aten._functional_assert_async.msg(condition, message, token)

        def op_calc_1(condition, message, token):
            y = torch.ops.aten._functional_assert_async.msg(condition, message, token)
            return y + y
        token = self._generate_tensor(shape, dtype, min=-2, max=2)
        condition = self._generate_tensor((), dtype, min=1, max=2)
        if dtype == 'bool':
            condition.fill_(True)
        message = 'aten assertion condition must be true'
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, condition, message, token,
        )
```
