# aten.glu.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.glu.default(self: Tensor, dim: int = -1) -> Tensor
```
### 1.2 算子功能
`aten.glu.default` 实现了**门控线性单元（Gated Linear Unit, GLU）** 激活函数。其核心计算过程如下：

1. 将输入张量 `input` 沿指定维度 `dim` 分割成形状相等的两半 `a` 和 `b`。
2. 对后半部分 `b` 应用 Sigmoid 激活函数：`σ(b)`。
3. 将 `a` 与 `σ(b)` 逐元素相乘（Hadamard 积），得到最终输出。

数学表达式为：
$$\text{GLU}(a, b) = a \otimes \sigma(b)$$

其中 `a` 是输入张量的前半部分，`b` 是后半部分，`σ` 是 Sigmoid 函数，`⊗` 表示逐元素乘积。
## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`/`input` | `Tensor` | 输入张量，须为浮点类型 |
| `dim` | `int` | 指定沿哪个维度将输入分割成两半。默认值：`-1`（即最后一个维度） |
| 返回值 | `Tensor` | 新的张量，形状与输入一致，但 `dim` 维度的大小减半 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | ×    |  -     |  -    |  -    | ×     | -      | ×      | √     | √      | √      | ×      |
| NPU    | -    | ×    |  -     |  -    |  -    | ×     | -      | ×      | √     | √      | √      | ×      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、切分`dim`要求为偶数

### 2.3 特殊限制说明

> 无


### 2.4 使用方法


```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_glu_default(self, shape, dtype):
        def op_calc(tensor):
            y = torch.ops.aten.glu.default(tensor, dim = 0)
            return y
        def op_calc_1(tensor):
            y = torch.ops.aten.glu.default(tensor, dim = 0)
            return y + y
        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor)
```
