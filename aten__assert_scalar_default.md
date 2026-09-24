# aten._assert_scalar.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._assert_scalar.default(Scalar self, str assert_msg) -> ()
```

### 1.2 算子功能

检查条件是否为真。条件为假时报告错误。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `number` | 输入张量或标量。 |
| `assert_msg` | `str` | 断言消息。 |
| `返回值` | `None` | 无返回值。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | x | x | x | √ |
| NPU | - | √ | - | - | - | √ | - | √ | x | x | x | √ |

float类型报错'float' object has no attribute 'meta'与社区版pytorch一致。
#### 2.2.2 Shape 支持

该算子不涉及 Tensor shape。

### 2.3 特殊限制说明

仅覆盖成功断言；失败异步断言可能影响后续设备任务。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'bool',]

    @parametrize('dtype', dtypes)
    def test_aten__assert_scalar_default(self, dtype):

        def op_calc(condition, message, token):
            torch.ops.aten._assert_scalar.default(condition, message)
            return token + token

        def op_calc_1(condition, message, token):
            torch.ops.aten._assert_scalar.default(condition, message)
            y = token + token
            return y + y

        token = torch.zeros(64, dtype=torch.float32, device=torch.device("npu"))
        condition = self._generate_tensor((), dtype, min=1, max=2)
        if dtype == 'bool':
            condition.fill_(True)
        condition = condition.item()
        message = 'aten assertion condition must be true'
        self.execute_inductor_test(
            (64,), dtype, op_calc, op_calc_1, condition, message, token,
        )
```
