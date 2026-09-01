# aten.min.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.min.out(
    Tensor self, 
    Tensor other,
    *, 
    Tensor(a!) out
) -> Tensor(a!)
```
### 1.2 算子功能
`aten.min.out` 是**逐元素比较**算子，对输入张量 `self` 和 `other` 逐元素比较，取较小值，并将结果写入 `out`：

1. **逐元素比较**：对 `self` 和 `other` 中每个位置的元素进行比较，取较小值。
2. **广播机制**：支持广播（broadcast），两个张量形状不必完全一致，但必须可广播。
3. **NaN 处理**：若某位置任一元素为 NaN，则结果为 NaN（NaN 在比较中视为特殊值）。
4. **输出形状**：结果为广播后的形状，与输入形状相关。
5. **out 变体语义**：结果不新建张量而是写入 `out` 并返回它；`out` 的形状、dtype 需与广播后的结果一致。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `other` | `Tensor` | 第二个输入张量 |
| `out` | `Tensor(a!)` | 输出张量（原地写入），0 维标量，dtype 与 `self` 一致 |
| 返回值 | `Tensor(a!)` | 即 `out`，全局最小值 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ | √ |
| NPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、shape 维度范围：[1-4]

2、 输出形状：广播后的形状，与输入形状相关（非固定标量）。

### 2.3 特殊限制说明

1. **输出张量 `out` 必须预先分配**：`out` 必须存在且形状、dtype 与广播后的结果一致。
2. **dtype 一致性**：`self`、`other`、`out` 三者的数据类型必须完全一致。
3. **别名安全性**：避免 `out` 与 `self` 或 `other` 共享内存（重叠），否则可能产生未定义行为

### 2.4 使用方法

```python
    dtypes = ['float32', 'float16', 'bfloat16', 'int8', 'int32', 'int64', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_min_out(self, shape, dtype):
        a = self._generate_tensor(shape, dtype)
        b = self._generate_tensor(shape, dtype)

        def op_calc(a, b):
            out = torch.empty_like(a)
            torch.ops.aten.min.out(a, b, out=out)
            return out

        def op_calc_1(a, b):
            out = torch.empty_like(a)
            torch.ops.aten.min.out(a, b, out=out)
            return out + out  

        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, a, b)    
```
