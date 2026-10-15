# aten.__irshift__.Scalar

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.__irshift__.Scalar(self: Tensor(a!), other: Scalar) -> Tensor(a!)
```
### 1.2 算子功能
1、`aten.__irshift__.Scalar` 是 `>>=` 运算符在 `other` 为标量时的原地版本：对 `self` 中每个整数按位算术右移 `other` 位（有符号整型高位补符号位），结果直接写回 `self`。
2、不分配新张量，返回修改后的 `self` 本身。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor(a!)` | 源张量，须为整型，**会被原地修改** |
| `other` | `Scalar` | 位移量，须小于 `self` 位宽 |
| 返回值 | `Tensor(a!)` | 右移后的 `self` 本身 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | ×      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | ×     | ×      | ×      | ×      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> 位移量须小于 `self` 元素位宽（int8<8、int32<32、int64<64），否则为未定义行为（UB）。


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten___irshift___Scalar(self, shape, dtype):
        def op_calc(tensor, scalar):
            y = torch.ops.aten.__irshift__.Scalar(tensor, scalar)
            return y

        def op_calc_1(tensor, scalar):
            y = torch.ops.aten.__irshift__.Scalar(tensor, scalar)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        scalar = 2
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, scalar)
```
