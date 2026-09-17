# aten.logical_or_

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.logical_or_(Tensor(a!) self, Tensor other) -> Tensor(a!)
```
### 1.2 算子功能
1、`aten.logical_or_` 是 `aten.logical_or` 的原地版本：逐元素逻辑或（非零为真），结果直接写回 `self`。

2、不分配新张量，返回修改后的 `self` 本身；结果 dtype 跟随 `self`（int8 进 int8 出，值为 0/1），不是 `bool`。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor(a!)` | 源张量，会被原地修改 |
| `other` | `Tensor` | 第二个源张量，与 `self` 广播对齐 |
| 返回值 | `Tensor(a!)` | 返回修改后的 `self` 本身 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

> 表中为输入 `self` 的 dtype；结果写回 `self`，dtype **跟随输入**。

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`self` 与 `other` 按 PyTorch 广播规则对齐；用例中两者同 shape。


### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_logical_or_(self, shape, dtype):
        def op_calc(first_element, second_element):
            y = torch.ops.aten.logical_or_(first_element, second_element)
            return y

        def op_calc_1(first_element, second_element):
            # 原地算两次: 同类图、多一个节点(logical_or 幂等, 结果与算一次相同)
            y = torch.ops.aten.logical_or_(first_element, second_element)
            y = torch.ops.aten.logical_or_(y, second_element)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, second_element)
```
