# aten.sigmoid.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sigmoid.default(Tensor self) -> Tensor
```

### 1.2 算子功能

`aten.sigmoid.default` 对输入张量逐元素计算 Sigmoid：`out[i] = 1 / (1 + exp(-self[i]))`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| 返回值 | `Tensor` | 与 self 同 shape 的结果张量 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |
| NPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、shape 维度范围：[1-4]，具体实测 shape 以框架配置为准。

2、输出 shape 与 `self` 相同。

### 2.3 特殊限制说明

> 整数输入返回默认浮点类型（通常为 fp32），浮点输入保持 dtype。bool 输入可通过类型提升执行，但不纳入本项目支持规格。

### 2.4 使用方法

`_generate_tensor` 保持输入 dtype 和设备，按 `min=-4, max=4` 生成小范围测试数据，整数取值为 `-4～3`。

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_sigmoid_default(self, shape, dtype):
        def op_calc(tensor):
            y = torch.ops.aten.sigmoid.default(tensor)
            return y

        def op_calc_1(tensor):
            y = torch.ops.aten.sigmoid.default(tensor)
            return y + y

        tensor = self._generate_tensor(shape, dtype, min=-4, max=4)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor)
```
