# aten.log2.out

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.log2.out(self: Tensor, *, out: Tensor(a!)) -> Tensor(a!)
```

### 1.2 算子功能

`aten.log2.out` 对输入张量 `self` 逐元素计算以 2 为底的对数，并将结果写入预先分配的 `out` 张量：

`out[i] = log2(self[i])`

返回值与 `out` 是同一个张量，具有相同的 dtype、shape、device 和底层存储。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量。当前支持声明覆盖 int8、int32、int64、fp16、fp32 和 bf16 输入 |
| `out` | `Tensor(a!)` | 预先分配的写回张量。当前测试配置使用 fp32，shape 和 device 必须与 `self` 一致 |
| 返回值 | `Tensor(a!)` | 写入结果后的 `out`，与传入的 `out` 别名 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

下表表示 `self` 的输入 dtype，当前已验证组合均使用 fp32 `out`。

| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | - |

#### 2.2.2 Shape 支持

1、`self` 的 shape 维度范围：[1-4]。

2、当前支持配置要求 `out.shape == self.shape`，输出 shape 与输入 shape 相同。

### 2.3 特殊限制说明

> 当前 fp16、bf16 输入按原 dtype 计算后写入 fp32 `out`，整数输入转换为 fp32 后计算。`out` 必须与 `self` 的 shape 和 device 一致，返回值与 `out` 别名。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_log2_out(self, shape, dtype):
        def op_calc(first_element, out_t):
            result = torch.ops.aten.log2.out(first_element, out=out_t)
            return result

        def op_calc_1(first_element, out_t):
            result = torch.ops.aten.log2.out(first_element, out=out_t)
            return result + 1

        first_element = self._generate_tensor(shape, dtype)
        first_element = first_element.abs().clamp_min(1)
        out_t = torch.empty(shape, dtype=torch.float32, device='npu')

        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, out_t)
```
