# aten.bitwise_xor

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.bitwise_xor(self: Tensor, other: Tensor) -> Tensor
```
### 1.2 算子功能
1、`aten.bitwise_xor` 对 `self` 与 `other` 逐元素按位异或；bool 输入按逻辑异或（`a != b`）处理。

2、实参为两个 Tensor，运行时解析到 `aten.bitwise_xor.Tensor`；输出 dtype 与输入一致（bool 进 bool 出）。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 源张量，须为整型或 bool |
| `other` | `Tensor` | 第二个操作数，须与 `self` 同 dtype 且形状相同 |
| 返回值 | `Tensor` | 新张量，dtype 与输入一致，形状与输入相同 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | √      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`other` 须与 `self` 形状相同，输出形状与输入相同（本批用例未覆盖广播）。

### 2.3 特殊限制说明

1、 仅支持整数与 bool：float/bf16/fp16 输入在 eager 报 `NotImplementedError: "bitwise_xor_cpu" not implemented for 'Half'/'BFloat16'/'Float'`。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_bitwise_xor(self, shape, dtype):
        def op_calc(first_element, second_element):
            y = torch.ops.aten.bitwise_xor(first_element, second_element)
            return y

        def op_calc_1(first_element, second_element):
            y = torch.ops.aten.bitwise_xor(first_element, second_element)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, second_element)
```
