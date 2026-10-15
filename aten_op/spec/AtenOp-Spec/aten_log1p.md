# aten.log1p

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.log1p(Tensor self) -> Tensor
```

### 1.2 算子功能

`aten.log1p` 对输入张量 `self` 逐元素计算 `ln(1 + self[i])`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| 返回值 | `Tensor` | 与输入 shape 相同的结果张量 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |

bool 实测通过，但不纳入当前支持规格，按 `×` 标记。

#### 2.2.2 Shape 支持

已验证 shape（维度 1-4）：`(16,)`、`(16, 32)`、`(16, 32, 64)`、`(16, 32, 64, 128)`。

### 2.3 特殊限制说明

> 有限实数结果要求输入大于 -1。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_log1p(self, shape, dtype):
        def op_calc(first_element):
            y = torch.ops.aten.log1p(first_element)
            return y

        def op_calc_1(first_element):
            y = torch.ops.aten.log1p(first_element)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)
```
