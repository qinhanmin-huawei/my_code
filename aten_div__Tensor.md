# aten.div_.Tensor

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.aten.div_.Tensor(self: Tensor(a!), other: Tensor) -> Tensor(a!)
```

### 1.2 算子功能

`aten.div_.Tensor` 对 `self` 和 `other` 执行逐元素真除法，并将结果原地写回 `self`：

`self_i = self_i / other_i`

`other` 可以按照 PyTorch 广播规则扩展，但广播后的结果形状必须与 `self.shape` 完全相同，因为原地操作不能改变 `self` 的形状。返回值与修改后的 `self` 别名，具有相同的 dtype、shape、device 和底层存储。

该重载不包含 `rounding_mode`，语义等价于 `rounding_mode=None` 的真除法。带 `rounding_mode` 的原地 Tensor 重载是另一个算子 `aten.div_.Tensor_mode`，不属于本文档范围。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor(a!)` | 被除数和原地写回目标。调用后其数据被修改，返回值与其别名 |
| `other` | `Tensor` | 除数张量；可以广播到 `self`，但不能要求扩大 `self` |
| 返回值 | `Tensor(a!)` | 修改后的 `self`，dtype、shape、device 和底层存储均保持不变 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × |
| NPU | - | × | - | - | - | × | - | × | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、`self` 的 shape 维度范围：[1-4]。

2、`other` 必须可广播到 `self.shape`，并满足：

```text
torch.broadcast_shapes(self.shape, other.shape) == self.shape
```

3、输出 shape 始终等于 `self.shape`，原地操作不允许通过广播扩大 `self`。

支持示例：

```text
self.shape  = (4, 8, 16, 32)
other.shape = (1, 8, 1, 32)
result.shape = (4, 8, 16, 32)
```

不支持示例：

```text
self.shape  = (4, 1)
other.shape = (4, 8)
```

两者虽然可以广播到 `(4, 8)`，但这会扩大 `self`，因此不能原地写回。

### 2.3 特殊限制说明

1、该算子会修改 `self`。调用方若仍需使用原始被除数，必须在调用前显式保存或克隆。

2、上游语义不允许把真除法结果原地写回 bool/整数 `self`。当前 GPU/NPU 已验证 int8、int32、int64 不支持，其余 bool/整数类型尚未测试；不要因为 `other` 可以是整数，就把算子整体声明为支持整数原地输入。

3、浮点除数为 `0` 时遵循 IEEE 浮点语义：非零有限值除以零产生带符号的 `inf`，`0 / 0` 产生 `nan`，不会抛出整数除零异常。

4、返回值必须与 `self` 别名；实现不能以分配新输出张量替代原地写回。

5、需要梯度的叶子张量不能直接执行该原地操作；不安全的内部或部分内存重叠也应保持 PyTorch 的报错行为。

6、0 维、空张量、complex 和混合 dtype 组合不在当前 GPU/NPU 基础支持声明内，需单独验证后再扩展规格。

### 2.4 使用方法

```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_div__Tensor(self, shape, dtype):
        def op_calc(first_element, second_element):
            result = torch.ops.aten.div_.Tensor(first_element, second_element)
            return result

        def op_calc_1(first_element, second_element):
            result = torch.ops.aten.div_.Tensor(first_element, second_element)
            return result + 1

        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype)

        self.execute_inductor_test(
            shape,
            dtype,
            op_calc,
            op_calc_1,
            first_element,
            second_element,
        )
```
