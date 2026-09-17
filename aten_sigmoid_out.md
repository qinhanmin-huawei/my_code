# aten.sigmoid.out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
```

### 1.2 算子功能

`aten.sigmoid.out` 对输入张量逐元素计算 Sigmoid 并写入 `out`：`out[i] = 1 / (1 + exp(-self[i]))`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `out` | `Tensor(a!)` | 独立分配的输出张量，与 self 同 shape、同设备 |
| 返回值 | `Tensor(a!)` | 写入结果的 out，返回值与 out 别名 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |
| NPU | self | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |

本轮 `out` 配置：浮点输入使用输入 dtype，整数输入使用默认浮点类型（通常为 fp32）。

#### 2.2.2 Shape 支持

1、shape 维度范围：[1-4]，具体实测 shape 以框架配置为准。

2、输出 shape 与 `self` 相同。

### 2.3 特殊限制说明

> `out` 与 `self` 同设备，示例不覆盖输入输出存储重叠。bool 输入可通过类型提升执行，但不纳入本项目支持规格。

### 2.4 使用方法

`_generate_tensor` 保持输入 dtype 和设备，按 `min=-4, max=4` 生成小范围测试数据，整数取值为 `-4～3`。

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_sigmoid_out(self, shape, dtype):
        def op_calc(tensor, out):
            y = torch.ops.aten.sigmoid.out(tensor, out=out)
            return y

        def op_calc_1(tensor, out):
            y = torch.ops.aten.sigmoid.out(tensor, out=out)
            return y + y

        tensor = self._generate_tensor(shape, dtype, min=-4, max=4)
        out_dtype = tensor.dtype if tensor.is_floating_point() else torch.get_default_dtype()
        out = torch.empty_like(tensor, dtype=out_dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, out)
```
