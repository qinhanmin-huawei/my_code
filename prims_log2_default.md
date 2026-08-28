# prims.log2.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.prims.log2.default(Tensor self) -> Tensor
```
### 1.2 算子功能
`prims.log2.default` 是 prim 命名空间下的逐元素计算以 2 为底的对数运算：

1. **逐元素计算**：对输入张量 `self` 的每个元素计算 log₂(x)。
2. **特殊值处理**：输入为正数时返回正常对数值；输入为 `+inf` 时返回 `inf`；输入为 0 时返回 `-inf`；输入为负数或 NaN 时返回 `NaN`。
3. **输出类型**：结果保持浮点类型，形状与输入相同。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量（浮点类型） |
| 返回值 | `Tensor` | 与 `self` 形状相同的张量，包含逐元素 log₂ 结果 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ |  |
| NPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ |  |

#### 2.2.2 Shape 支持

1、shape 维度范围：[1-4]

2、输出张量的形状与输入张量相同

### 2.3 特殊限制说明

> 无

### 2.4 使用方法

```python
    dtypes = ['int8','int32','int64','float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_prims_log2_default(self, shape, dtype):
        def op_calc(first_element):
            x = first_element
            y = torch.ops.prims.log2.default(x)
            result = y
            return result
        def op_calc_1(first_element):
            x = first_element + 1
            y = torch.ops.prims.log2.default(x)
            result = y + 1
            return result
        first_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)
```
