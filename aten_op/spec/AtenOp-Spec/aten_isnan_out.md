# aten.isnan.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.isnan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
```
### 1.2 算子功能
1、`aten.isnan.out` 是 `aten.isnan.default` 的 `out` 版本：判断结果写进调用方提供的 `out`，并返回该 `out`。

2、`out` 必须是 `bool` 张量，形状与 `self` 一致。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量 |
| `out` | `Tensor` | 输出张量，须为 `bool`，形状与 `self` 一致 |
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

2、`out` 的形状与 `self` 一致。


### 2.3 特殊限制说明

1、 `out` 须按 `torch.bool` 分配，输出 dtype 与输入无关。

2、 整数/布尔没有 NaN 表示，结果恒为全 False；仅浮点可能出 True。

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_isnan_out(self, shape, dtype):
        def op_calc(first_element):
            y = torch.empty_like(first_element, dtype=torch.bool)
            torch.ops.aten.isnan.out(first_element, out=y)
            return y

        def op_calc_1(first_element):
            y = torch.empty_like(first_element, dtype=torch.bool)
            torch.ops.aten.isnan.out(first_element, out=y)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        if dtype in ('float16', 'bfloat16', 'float32'):
            flat = first_element.view(-1)
            flat[0] = float('nan')
            if flat.numel() > 1:
                flat[-1] = float('nan')
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element)
```
