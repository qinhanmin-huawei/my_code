# aten.sub.Tensor

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.sub.Tensor(self: Tensor, other: Tensor, *, alpha: Scalar = 1) -> Tensor
```
### 1.2 算子功能
1、`aten.sub.Tensor` 是逐元素（Element-wise）的减法算子：对输入张量 `self` 与 `other` 中对应元素执行减法 out = self - alpha × other，默认 `alpha = 1`。
2、`self` 与 `other` 支持广播（broadcasting），输出张量的形状与广播后的输入一致，dtype 与输入相同（同 dtype 输入时无类型提升）。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 被减数张量 |
| `other` | `Tensor` | 减数张量，须与 `self` 可广播（broadcastable） |
| `alpha` | `Scalar` | 减数的缩放系数，默认为 `1`，即 out = self - alpha × other |
| 返回值 | `Tensor` | 一个新的张量，形状与广播后的输入一致，dtype 与输入相同（同 dtype 输入时） |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`self` 与 `other` 支持广播，输出张量形状为广播后的形状


### 2.3 特殊限制说明


> 整型输入时 `alpha` 不能为浮点数，否则报错 `For integral input tensors, argument alpha must not be a floating point number`；布尔 `alpha` 仅允许结果为布尔时使用。



### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_sub_Tensor(self, shape, dtype):
        def op_calc(first_element, second_element):
            y = torch.ops.aten.sub.Tensor(first_element, second_element)
            return y

        def op_calc_1(first_element, second_element):
            y = torch.ops.aten.sub.Tensor(first_element, second_element)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        # 固定小标量避免 int8 结果溢出
        second_element = torch.full(shape, 2, dtype=first_element.dtype, device="npu")
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, second_element)
```
