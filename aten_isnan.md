# aten.isnan

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.isnan(Tensor self) -> Tensor
```
### 1.2 算子功能
1、`aten.isnan` 是 packet 调用形式，按实参解析到 `aten.isnan.default`：逐元素判断是否为 NaN，返回同形状的 bool 张量。

2、输出恒为 `bool`，与输入 dtype 无关。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| 返回值 | `Tensor` | bool 张量，形状与 `self` 一致 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

> 表中为输入 `self` 的 dtype；输出**恒为 `bool`**，与输入 dtype 无关。

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、输出形状与 `self` 一致（逐元素）。


### 2.3 特殊限制说明

1、 整数/布尔没有 NaN 表示，结果恒为全 False；仅浮点可能出 True。

2、 浮点用例须手动塞 NaN，否则输出恒为全 False、没有分辨力。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_isnan(self, shape, dtype):
        def op_calc(first_element):
            y = torch.ops.aten.isnan(first_element)
            return y

        def op_calc_1(first_element):
            y = torch.ops.aten.isnan(first_element)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        if dtype in ('float16', 'bfloat16', 'float32'):
            flat = first_element.view(-1)
            flat[0] = float('nan')
            if flat.numel() > 1:
                flat[-1] = float('nan')
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)
```
