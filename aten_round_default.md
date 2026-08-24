# aten.round.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.round.default(Tensor self) -> Tensor
```
### 1.2 算子功能
`aten.round.default` 对输入张量中的每个元素执行舍入取整，返回与输入形状相同的新张量。对于浮点数，采用四舍六入五成双（round half to even，即银行家舍入）规则：当小数部分恰好为 0.5 时，舍入到最近的偶数整数（如 0.5→0、1.5→2、2.5→2）；对于整数类型，直接返回原值。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| 返回值 | `Tensor` | 与 `self` 形状相同的张量，元素为舍入后的结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ | × |
| NPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、输出张量的形状与输入张量相同

### 2.3 特殊限制说明

> 无

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64','float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_round_default(self, shape, dtype):
        def op_calc(input):
            y = torch.ops.aten.round.default(input)
            return y
        def op_calc_1(input):
            y = torch.ops.aten.round.default(input)
            return y + y
        input = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input)
```
