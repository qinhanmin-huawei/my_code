# aten.logical_and.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.logical_and.default(Tensor self, Tensor other) -> Tensor
```
### 1.2 算子功能
1、`aten.logical_and.default` 是 `aten.logical_and` 的默认重载，签名与 packet 形式一致：逐元素逻辑与（非零为真），返回同形状的 `bool` 张量。

2、导出清单中该重载以 `.default` 形式单列，与 packet 形式各记一例。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 第一个源张量 |
| `other` | `Tensor` | 第二个源张量，与 `self` 广播对齐 |
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

2、`self` 与 `other` 按 PyTorch 广播规则对齐，输出形状为广播后的 shape；用例中两者同 shape。


### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_logical_and_default(self, shape, dtype):
        def op_calc(first_element, second_element):
            y = torch.ops.aten.logical_and.default(first_element, second_element)
            return y

        def op_calc_1(first_element, second_element):
            y = torch.ops.aten.logical_and.default(first_element, second_element)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, second_element)
```
