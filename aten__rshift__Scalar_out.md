# aten.__rshift__.Scalar_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.__rshift__.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
```
### 1.2 算子功能
1、`aten.__rshift__.Scalar_out` 将 `self` 中每个整数按位算术右移 `other` 位（有符号整型高位补符号位），结果直接覆盖写入 `out`。
2、`other` 为统一位移量，须小于 `self` 元素位宽；`out` 的 dtype/形状须与输入一致。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 源张量，须为整型 |
| `other` | `Scalar` | 位移量，须小于 `self` 位宽 |
| `out` | `Tensor` | 输出张量，dtype/形状须与 `self` 一致，内容会被覆盖 |
| 返回值 | `Tensor(a!)` | `out` 张量本身 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | ×      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | ×      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、输出张量与输入张量形状相同


### 2.3 特殊限制说明

> 位移量须小于 `self` 元素位宽（int8<8、int32<32、int64<64），否则为未定义行为（UB）。


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten___rshift___Scalar_out(self, shape, dtype):
        def op_calc(first_element, scalar):
            y = torch.empty_like(first_element)
            torch.ops.aten.__rshift__.Scalar_out(first_element, scalar, out=y)
            return y

        def op_calc_1(first_element, scalar):
            y = torch.empty_like(first_element)
            torch.ops.aten.__rshift__.Scalar_out(first_element, scalar, out=y)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        scalar = 2
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, scalar)
```
