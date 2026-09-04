# aten.div.Scalar_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.div.Scalar_out(self: Tensor, other: Scalar, *, out: Tensor) -> Tensor
```
### 1.2 算子功能
1、`aten.div.Scalar_out` 对输入张量 `self` 中的每个元素除以标量 `other`，执行逐元素真除法 out = self / other，并将结果直接覆盖写入调用方预先指定的 `out` 张量。
2、结果 dtype：int输入提升为 float32；浮点输入与输入同 dtype。因此 `out` 需按**结果 dtype**分配（int/bool 输入用 float32）。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 被除数张量 |
| `other` | `Scalar` | 除数标量（Python 整数或浮点数） |
| `out` | `Tensor` | 输出张量，须按结果 dtype 分配（int输入为 float32），device/形状需匹配，计算前其内容会被覆盖 |
| 返回值 | `Tensor` | `out` 张量本身 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |


#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、输出张量与输入张量形状相同


### 2.3 特殊限制说明

> 1、`out` 必须按结果 dtype 分配：int输入 → float32，浮点输入 → 同输入 dtype；否则 eager 报类型转换错误。
> 2、标量 `other` 为 0 时遵循 IEEE 浮点语义（得 ±inf/nan），不抛整数除零异常。


### 2.4 使用方法


```python
    def _div_result_dtype(dtype):
        if dtype in ('int8', 'int32', 'int64', 'bool'):
            return torch.float32
        return getattr(torch, dtype)

    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_div_Scalar_out(self, shape, dtype):
        def op_calc(first_element, scalar):
            out = torch.empty(shape, dtype=_div_result_dtype(dtype), device="npu")
            torch.ops.aten.div.Scalar_out(first_element, scalar, out=out)
            return out

        def op_calc_1(first_element, scalar):
            out = torch.empty(shape, dtype=_div_result_dtype(dtype), device="npu")
            torch.ops.aten.div.Scalar_out(first_element, scalar, out=out)
            return out + out

        first_element = self._generate_tensor(shape, dtype)
        scalar = self._generate_random_scalar(dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, scalar)
```
