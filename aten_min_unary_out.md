# aten.min.unary_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.min.unary_out(
    Tensor self, 
    *, 
    Tensor(a!) out
) -> Tensor(a!)
```
### 1.2 算子功能
`aten.min.unary_out` 是全局归约算子，求输入张量 `self` 中所有元素的**最小值**，并将结果原地写入 `out`：

1. **全局归约**：遍历 `self` 全部元素，取最小值。
2. **NaN 处理**：若 `self` 中存在 NaN，结果为 NaN（NaN 在比较中视为最小）。
3. **输出形状**：结果为 0 维标量张量，与输入形状无关。
4. **out 变体语义**：结果不新建张量，而是写入预先分配的 out 张量并返回它。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `out` | `Tensor(a!)` | 输出张量（原地写入），0 维标量，dtype 与 `self` 一致 |
| 返回值 | `Tensor(a!)` | 即 `out`，全局最小值 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、shape 维度范围：[1-4]

2、输出为 0 维标量张量，与输入形状无关

### 2.3 特殊限制说明

> `out` 必须为 0 维张量且 dtype 与 `self` 一致，否则报错。

### 2.4 使用方法

```python
    dtypes = ['float32', 'float16', 'bfloat16', 'int8', 'int32', 'int64', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_min_unary_out(self, shape, dtype):
        a = self._generate_tensor(shape, dtype)

        def op_calc(a):
            out = torch.tensor(0, dtype=eval('torch.' + dtype), device=torch.device("npu"))
            torch.ops.aten.min.unary_out(a, out=out)
            return out

        def op_calc_1(a):
            out = torch.tensor(0, dtype=eval('torch.' + dtype), device=torch.device("npu"))
            torch.ops.aten.min.unary_out(a, out=out)
            return out + out  
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, a)
```
