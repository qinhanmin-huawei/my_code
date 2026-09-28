# aten.eq.Scalar_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
```
### 1.2 算子功能
1、`aten.eq.Scalar_out` 是 `aten.eq.Scalar` 的 `out` 版本：比较结果写入调用方提供的 `out` 并返回该 `out`。

2、`out` 的形状与 `self` 一致。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 被比较的第一个张量 |
| `other` | `Scalar` | 比较用的标量，按 `self` 的 dtype 装箱 |
| `out` | `Tensor` | 输出张量，形状与 `self` 一致 |
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

2、输出形状与 `self` 一致（标量广播）。


### 2.3 特殊限制说明

1、 输出恒为 `bool`，与输入 dtype 无关。

2、 `out` 的 dtype 不做强制校验：实测传 int32 / float32 的 `out` 也静默成功（写出 0 / 1）；用例按比较结果的真值 dtype `torch.bool` 分配。

3、 `other` 是泛化 `Scalar`，比较前按 `self` 的 dtype 装箱。


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_eq_Scalar_out(self, shape, dtype):
        def op_calc(first_element, scalar):
            y = torch.empty_like(first_element, dtype=torch.bool)
            torch.ops.aten.eq.Scalar_out(first_element, scalar, out=y)
            return y

        def op_calc_1(first_element, scalar):
            y = torch.empty_like(first_element, dtype=torch.bool)
            torch.ops.aten.eq.Scalar_out(first_element, scalar, out=y)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        scalar = self._generate_random_scalar(dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, scalar)
```
