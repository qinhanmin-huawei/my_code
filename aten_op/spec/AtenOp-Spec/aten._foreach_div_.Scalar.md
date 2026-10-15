# aten._foreach_div_.Scalar

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten._foreach_div_.Scalar(self: Tensor(a!)[], scalar: Scalar) -> ()
```

### 1.2 算子功能

将输入 Tensor 列表中的元素逐项执行原地除法。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor[]` | 被原地修改的被除数列表。 |
| `scalar` | `Scalar` | 固定除数。 |
| `返回值` | `()` | 算子原地修改 `self`。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | - | - | - | - | - | - | - | √ | √ | √ | - |
| NPU | - | - | - | - | - | - | - | - | √ | √ | √ | - |

#### 2.2.2 Shape 支持

输入列表中的 Tensor 支持 1～4 维。

### 2.3 特殊限制说明

除数必须非零。整数输入执行真除法后结果会提升为 float32，而该算子需要将结果原地写回输入列表中的整数 Tensor，浮点结果无法安全转换并写回原 dtype，因此不支持 int8、int32 和 int64。bool 对真除法无实际意义；不覆盖复数。

### 2.4 使用方法

```python
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten__foreach_div__scalar(self, shape, dtype):
        def op_calc(tensors):
            torch.ops.aten._foreach_div_.Scalar(tensors, 2.0)
            return tensors

        def op_calc_1(tensors):
            torch.ops.aten._foreach_div_.Scalar(tensors, 2.0)
            return [item + item for item in tensors]

        tensors = [self._generate_tensor(shape, dtype, min=1, max=8) for _ in range(2)]
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensors)
```
