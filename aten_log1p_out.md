# aten.log1p.out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.log1p.out(self: Tensor, *, out: Tensor(a!)) -> Tensor(a!)
```

### 1.2 算子功能

`aten.log1p.out` 对 `self` 逐元素计算 `ln(1 + self[i])`，并将结果写入 `out`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `out` | `Tensor(a!)` | 预先分配的写回张量 |
| 返回值 | `Tensor(a!)` | 写入结果后的 `out` |

### 2.2 支持规格

#### 2.2.1 DataType 支持

下表表示 `self` 的输入 dtype，当前测试配置使用 fp32 `out`。

| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |

bool 实测通过，但不纳入当前支持规格，按 `×` 标记。

#### 2.2.2 Shape 支持

已验证 shape（维度 1-4）：`(16,)`、`(16, 32)`、`(16, 32, 64)`、`(16, 32, 64, 128)`。

### 2.3 特殊限制说明

> 有限实数结果要求输入大于 -1。当前测试要求 `out` 为 fp32，且与 `self` 的 shape 和 device 一致；返回值与 `out` 别名。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_log1p_out(self, shape, dtype):
        def op_calc(first_element, out):
            y = torch.ops.aten.log1p.out(first_element, out=out)
            return y

        def op_calc_1(first_element, out):
            y = torch.ops.aten.log1p.out(first_element, out=out)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        out = torch.empty(shape, dtype=torch.float32, device='npu')
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, out)
```
