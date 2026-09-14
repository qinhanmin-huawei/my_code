# aten.erf.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.erf.out(self: Tensor, *, out: Tensor) -> Tensor
```
### 1.2 算子功能
1、`aten.erf.out` 对输入张量 `self` 中的每个元素 x 计算高斯误差函数 y = erf(x)，并将结果直接覆盖写入调用方预先指定的 `out` 张量。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 参与计算的源张量 |
| `out` | `Tensor` | 输出张量，dtype 与浮点输入一致，device/形状需匹配，计算前其内容会被覆盖 |
| 返回值 | `Tensor` | `out` 张量本身 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

>注：aclnn不支持int8，int32。

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、输出张量与输入张量形状相同


### 2.3 特殊限制说明

无

### 2.4 使用方法


```python
    dtypes = ['int64', 'bool', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_erf_out(self, shape, dtype):
        def op_calc(first_element):
            out_dtype = first_element.dtype if first_element.is_floating_point() else torch.float32
            out = torch.zeros_like(first_element, dtype=out_dtype)
            torch.ops.aten.erf.out(first_element, out=out)
            return out

        def op_calc_1(first_element):
            out_dtype = first_element.dtype if first_element.is_floating_point() else torch.float32
            out = torch.zeros_like(first_element, dtype=out_dtype)
            torch.ops.aten.erf.out(first_element, out=out)
            return out + out

        first_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)
```
