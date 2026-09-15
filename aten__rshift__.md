# aten.__rshift__

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.__rshift__(Tensor self, Tensor other) -> Tensor
```
### 1.2 算子功能
1、`aten.__rshift__` 是 `>>` 运算符在 `other` 为张量时的重载：对 `self` 中每个整数按位算术右移 `other` 对应元素的位数（有符号整型高位补符号位），返回新张量。
2、`other` 与 `self` 按广播对齐。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 源张量，须为整型 |
| `other` | `Tensor` | 位移量张量，每个元素须小于 `self` 位宽，可与 `self` 广播 |
| 返回值 | `Tensor` | 新张量，形状为广播后的形状 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | ×      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | ×      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、`other` 与 `self` 支持广播，输出形状为广播后的形状


### 2.3 特殊限制说明

> 位移量须小于 `self` 元素位宽（int8<8、int32<32、int64<64），否则为未定义行为（UB）。


### 2.4 使用方法


```python
    def _generate_shift_tensor(self, shape, dtype):
        shift_max = {'int8': 8, 'int32': 32, 'int64': 64}[dtype]
        return self._generate_tensor(shape, dtype, max=shift_max)

    dtypes = ['int8', 'int32', 'int64']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten___rshift__(self, shape, dtype):
        def op_calc(first_element, second_element):
            y = torch.ops.aten.__rshift__(first_element, second_element)
            return y

        def op_calc_1(first_element, second_element):
            y = torch.ops.aten.__rshift__(first_element, second_element)
            return y + y

        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_shift_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, first_element, second_element)
```
