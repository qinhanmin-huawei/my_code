# aten.glu.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.glu.out(
    self: Tensor, 
    dim: int = -1, 
    *, 
    out: Tensor
) -> Tensor
```
### 1.2 算子功能
`aten.glu.out` 的核心逻辑主要包含以下三个步骤：

1. **张量切分**：将输入张量（`input`）沿着指定的维度（`dim`）平均切分为两半，分别记为 $a$ 和 $b$。因此，输入张量在该维度上的长度必须是偶数。
2. **门控计算**：将后半部分 $b$ 通过 Sigmoid 激活函数（$\sigma$）转换为门控权重，并与前半部分 $a$ 进行逐元素相乘（Hadamard Product）。
3. **输出重定向**：最终的计算结果会直接覆盖或写入到预先指定的 `out` 张量中。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self/input`        | `Tensor`          | 参与计算的源张量 |
| `dim`       | `int `    | 执行切分和计算的轴/维度|
| `out `      | `Tensor`| 其数据类型（Dtype）和设备（Device）必须与输入张量匹配。其形状必须与预期输出的形状严格一致（即除了 dim 维度减半外，其余维度与 input 相同） |


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

以下示例实现了原子异或计算：

```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (128, 128, 1024, 4096)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_glu_out(self, shape, dtype):
        def op_calc(input, out_t):
            y = torch.ops.aten.glu.out(input, dim = 0, out = out_t)
            return out_t
        def op_calc_1(input, out_t):
            y = torch.ops.aten.glu.out(input, dim = 0, out = out_t)
            return out_t + out_t
        out_shape  = (shape[0] // 2,) + shape[1:]
        out_t = torch.empty(out_shape,  device='npu', dtype = eval("torch."+dtype))
        input = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input, out_t)
```
