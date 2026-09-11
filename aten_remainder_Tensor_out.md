# aten.remainder.Tensor_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.remainder.Tensor_out(self: Tensor, other: Tensor, *, out: Tensor) -> Tensor
```


### 1.2 算子功能
`aten.remainder.Tensor_out` 计算输入张量 `self` 与 `other` 之间的**逐元素取余（remainder）** 运算，并将结果存入 `out` 张量。

- 对于整数张量：执行整数取余，结果符号与被除数（即 `self` 的元素）相同。
- 对于浮点张量：执行浮点取余，结果为 `self - other * floor(self / other)`，符号与被除数相同。
- 两个输入张量必须可广播（broadcastable）至相同形状。
- **输出重载特性**：不会分配新张量，直接修改 `out`，减少内存分配开销。
## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 输入张量（被除数），支持浮点或整型 |
| `other` | `Tensor` | 除数张量，须与 `self` 可广播且类型兼容，元素不能为零（整数零报错，浮点零产生 inf 或 nan） |
| `out` | `Tensor` | 用于保存输出结果的张量，形状须与广播后的形状一致，数据类型须与结果类型兼容，该张量会被原地修改 |
| 返回值 | `Tensor` | 返回 `out` 张量本身（即计算结果） |
### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | ×    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |
| NPU    | -    | ×    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['int32', 'int64','float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]
    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_remainder_tensor_out(self, shape, dtype):
        def op_calc(first_element, second_element, out):
            torch.ops.aten.remainder.Tensor_out(first_element, second_element, out=out)
            return out
        def op_calc_1(first_element, second_element, out):
            torch.ops.aten.remainder.Tensor_out(first_element, second_element, out=out)
            result = out + 1
            return result
        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype)
        if dtype != "float16" and dtype != "bfloat16":
            out = torch.empty_like(first_element,dtype=torch.float32)
        else:
            out = torch.empty_like(first_element)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, second_element, out)
```
