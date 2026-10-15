# aten.bitwise_and.Scalar_Tensor_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.bitwise_and.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
```
### 1.2 算子功能
1、`aten.bitwise_and.Scalar_Tensor_out` 是标量在前、结果写入 `out` 的版本：标量 `self` 广播到 `other` 的形状后逐元素按位与。

2、`out` 的 dtype 须跟随 `other`：bool 张量配 `True` 时 `out` 为 bool。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self` | `Scalar` | 参与按位与的标量，须与 `other` 的 dtype 匹配（bool 档传 `True`） |
| `other` | `Tensor` | 源张量，须为整型或 bool |
| `out` | `Tensor` | 输出张量，dtype/形状须与 `other` 一致，内容会被覆盖 |
| 返回值 | `Tensor(a!)` | `out` 张量本身 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | √      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`out` 形状须与 `other` 相同，内容会被覆盖。

### 2.3 特殊限制说明

1、 仅支持整数与 bool：float/bf16/fp16 输入在 eager 报 `NotImplementedError: "bitwise_and_cpu" not implemented for 'Half'/'BFloat16'/'Float'`。

2、 结果 dtype 跟随 `other`：bool 张量须配 Python `True`，`out` 须与 `other` 同 dtype。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_bitwise_and_Scalar_Tensor_out(self, shape, dtype):
        def op_calc(second_element, scalar):
            y = torch.empty_like(second_element)
            torch.ops.aten.bitwise_and.Scalar_Tensor_out(scalar, second_element, out=y)
            return y

        def op_calc_1(second_element, scalar):
            y = torch.empty_like(second_element)
            torch.ops.aten.bitwise_and.Scalar_Tensor_out(scalar, second_element, out=y)
            return y + y

        second_element = self._generate_tensor(shape, dtype)
        # bool 档须传 Python True：否则结果提升为 int64，与 empty_like 的 bool 对不上
        scalar = True if dtype == 'bool' else 5
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, second_element, scalar)
```
