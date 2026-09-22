# aten.bitwise_and.Tensor_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
```
### 1.2 算子功能
1、`aten.bitwise_and.Tensor_out` 与 `aten.bitwise_and.Tensor` 同计算：逐元素按位与，结果直接覆盖写入 `out`。

2、`out` 的 dtype/形状须与输入一致（bool 档的 `out` 也须为 bool）。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 源张量，须为整型或 bool |
| `other` | `Tensor` | 第二个操作数，须与 `self` 同 dtype 且形状相同 |
| `out` | `Tensor` | 输出张量，dtype/形状须与输入一致，内容会被覆盖 |
| 返回值 | `Tensor(a!)` | `out` 张量本身 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | √      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`out` 形状须与输入相同，内容会被覆盖（本批用例未覆盖广播）。

### 2.3 特殊限制说明

1、 仅支持整数与 bool：float/bf16/fp16 输入在 eager 报 `NotImplementedError: "bitwise_and_cpu" not implemented for 'Half'/'BFloat16'/'Float'`。

2、 `out` 的 dtype 须与输入一致。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_bitwise_and_Tensor_out(self, shape, dtype):
        def op_calc(first_element, second_element):
            y = torch.empty_like(first_element)
            torch.ops.aten.bitwise_and.Tensor_out(first_element, second_element, out=y)
            return y

        def op_calc_1(first_element, second_element):
            y = torch.empty_like(first_element)
            torch.ops.aten.bitwise_and.Tensor_out(first_element, second_element, out=y)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, second_element)
```
