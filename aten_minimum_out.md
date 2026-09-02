# aten.minimum.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.minimum.out(
    Tensor self, 
    Tensor other, 
    *, 
    Tensor(a!) out
) -> Tensor(a!)
```
### 1.2 算子功能
`aten.minimum.out` 是逐元素二元算子，对输入张量 `self` 与 `other` 的每个对应元素取**最小值**，并将结果原地写入 `out`：

1. **逐元素比较**：`out_i = min(self_i, other_i)`，结果形状为 `self` 与 `other` 广播后的形状。
2. **广播规则**：`self` 与 `other` 形状无需相同，但必须可广播；`out` 形状需与广播结果一致。
3. **NaN 处理**：当参与比较的元素之一为 NaN 时，返回该 NaN。
4. **out 变体语义**：结果不新建张量而是写入 `out` 并返回它；`out` 的 dtype 需与输入一致。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `other` | `Tensor` | 第二个输入张量，需与 `self` 可广播 |
| `out` | `Tensor(a!)` | 输出张量（原地写入），形状为广播后形状，dtype 与输入一致 |
| 返回值 | `Tensor(a!)` | 即 `out`，逐元素最小值 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ | √ |
| NPU |  | √ |  |  |  | √ |  | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、shape 维度范围：[1-4]

2、输出张量形状为 `self` 与 `other` 广播后的形状；测试中两者同形，故 `out` 与输入同形

### 2.3 特殊限制说明

> 不支持复数类型；`out` 的形状需与广播结果一致、dtype 与输入一致；当元素为 NaN 时返回 NaN。

### 2.4 使用方法

```python
    dtypes = ['int8','int32','int64','float16', 'bfloat16', 'float32','bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_aten_minimum_out(self, shape, dtype):
        a = self._generate_tensor(shape, dtype)
        b = self._generate_tensor(shape, dtype)

        def op_calc(a, b):
            out = torch.empty_like(a)
            torch.ops.aten.minimum.out(a, b, out=out)
            return out

        def op_calc_1(a, b):
            out = torch.empty_like(a)
            torch.ops.aten.minimum.out(a, b, out=out)
            return out + out

        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, a, b)
```
