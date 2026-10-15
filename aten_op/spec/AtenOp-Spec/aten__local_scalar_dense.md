# aten._local_scalar_dense

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._local_scalar_dense(Tensor self) -> Scalar
```

### 1.2 算子功能

读取单元素 Tensor 的数值并返回 Python 标量。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量或标量。 |
| `返回值` | `number` | 算子计算结果。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

输入 Tensor 支持 0～1 维。

### 2.3 特殊限制说明

输入仅含一个元素。标量由辅助 Tensor 承接；capture_scalar_outputs 开启后仍需目标后端支持数据相关标量读取。

### 2.4 使用方法

```python
    shapes = [(), (1,)]
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__local_scalar_dense(self, shape, dtype):
        scalar_capture = torch._dynamo.config.patch(capture_scalar_outputs=True)
        scalar_capture.__enter__()
        self.addCleanup(scalar_capture.__exit__, None, None, None)

        def op_calc(first_element, token):
            scalar = torch.ops.aten._local_scalar_dense(first_element)
            return token + scalar

        def op_calc_1(first_element, token):
            scalar = torch.ops.aten._local_scalar_dense(first_element)
            y = token + scalar
            return y + y
        tensor1 = self._generate_tensor(shape, dtype, min=-2, max=2)
        token = torch.zeros(64, dtype=torch.float32, device=torch.device("npu"))
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, tensor1, token,
        )
```
