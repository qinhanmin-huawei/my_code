# aten.lt.Tensor_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
```
### 1.2 算子功能
1、`aten.lt.Tensor_out` 是 `aten.lt.Tensor` 的 `out` 版本：逐元素比较结果写入调用方提供的 `out` 并返回该 `out`。

2、`out` 必须是 `bool` 张量，形状为广播后的 shape。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 被比较的第一个张量 |
| `other` | `Tensor` | 第二个比较数，与 `self` 广播对齐 |
| `out` | `Tensor` | 输出张量，须为 `bool`，形状为广播后的 shape |
| 返回值 | `Tensor` | bool 张量，形状为广播后的 shape |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

> 表中为输入 `self` / `other` 的 dtype；输出**恒为 `bool`**，与输入 dtype 无关。

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`self` 与 `other` 按 PyTorch 广播规则对齐，`out` 的形状为广播后的 shape；用例中两者同 shape。


### 2.3 特殊限制说明

> 无

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_lt_Tensor_out(self, shape, dtype):
        def op_calc(first_element, second_element):
            y = torch.empty_like(first_element, dtype=torch.bool)
            torch.ops.aten.lt.Tensor_out(first_element, second_element, out=y)
            return y

        def op_calc_1(first_element, second_element):
            y = torch.empty_like(first_element, dtype=torch.bool)
            torch.ops.aten.lt.Tensor_out(first_element, second_element, out=y)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, second_element)
```
