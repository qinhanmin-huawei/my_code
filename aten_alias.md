# aten.alias

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.alias.default(self: Tensor) -> Tensor
```
### 1.2 算子功能
`aten.alias` 返回与输入张量共享相同底层存储的视图（View）张量。对返回张量的修改会反映到原始张量上，反之亦然。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 源张量 |
| 返回值 | `Tensor` | 与 `self` 共享存储的别名视图张量，形状与 `self` 相同 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |


#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

### 2.3 特殊限制说明

> 无

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_alias(self, shape, dtype):
        def op_calc(input):
            y = torch.ops.aten.alias(input)
            return y
        def op_calc_1(input):
            y = torch.ops.aten.alias(input)
            return y + y
        input = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input)
```