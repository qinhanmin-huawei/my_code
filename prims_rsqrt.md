# prims.rsqrt

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.prims.rsqrt(Tensor self) -> Tensor
```
### 1.2 算子功能
`prims.rsqrt` 对输入张量中的每个元素计算平方根的倒数，即 $1 / \sqrt{x}$。对于负数和零，行为取决于具体实现（通常零返回 `inf`，负数返回 `nan`）。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| 返回值 | `Tensor` | 与 `self` 形状相同的张量，元素为 $1 / \sqrt{x}$ 的结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | × | × | × | × | × | × | × | × | √ | √ | √ | × |
| NPU | × | × | × | × | × | × | × | × | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、输出张量的形状与输入张量相同

### 2.3 特殊限制说明

> 无

### 2.4 使用方法

```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_prims_rsqrt(self, shape, dtype):
        def op_calc(input):
            y = torch.ops.prims.rsqrt(input)
            return y
        def op_calc_1(input):
            y = torch.ops.prims.rsqrt(input)
            return y + y
        input = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input)
```
