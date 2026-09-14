# aten.avg_pool2d

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.avg_pool2d(
    self: Tensor,
    kernel_size: int[2],
    stride: int[2] = [],
    padding: int[2] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int? = None
) -> Tensor
```
### 1.2 算子功能
`aten.avg_pool2d` 的核心功能是对输入张量 `self` 执行 2D 平均池化：使用大小为 `kernel_size` 的滑动窗口按 `stride` 步长在空间维度 `H`、`W` 上滑动，对每个窗口内的元素求平均值作为输出。`padding` 用于在空间边界补零，`ceil_mode` 决定输出尺寸的取整方式，`count_include_pad` 决定求均值时是否将补零元素计入分母，`divisor_override` 可显式指定除法分母。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 输入张量，布局为 4D `(N, C, H, W)` 或 3D `(C, H, W)` |
| `kernel_size` | `int[2]` | 池化窗口大小 `(kh, kw)` |
| `stride` | `int[2]` | 滑动步长，默认为空（此时等于 `kernel_size`） |
| `padding` | `int[2]` | 空间维度两侧补零数量，默认为 0 |
| `ceil_mode` | `bool` | 是否向上取整计算输出尺寸，默认为 False |
| `count_include_pad` | `bool` | 求均值时是否把补零元素计入分母，默认为 True |
| `divisor_override` | `int?` | 显式指定除法分母，默认为 None |
| 返回值 | `Tensor` | 池化后的输出张量，布局与输入一致（4D `(N, C, H_out, W_out)` 或 3D `(C, H_out, W_out)`） |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、shape维度范围：[3-4]（3D `(C, H, W)` 或 4D `(N, C, H, W)`，3D 时按单样本 N=1 处理）

2、`H`、`W` 需大于等于 `kernel_size`，输出空间尺寸满足：`H_out = floor((H + 2*padding - kernel_size) / stride) + 1`（`ceil_mode=False` 时）

### 2.3 特殊限制说明

> `kernel_size`、`stride`、`padding` 为长度为 2 的列表，分别对应 `H`、`W` 两个维度。使用 ChannelsLast 内存格式时仅支持 4D 输入。当 `ceil_mode=True` 时，输出尺寸向上取整。

### 2.4 使用方法

```python
    dtypes = ['float16', 'bfloat16', 'float32', 'int8', 'int32', 'int64']
    shapes = [(128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_avg_pool2d(self, shape, dtype):
        def op_calc(input):
            y = torch.ops.aten.avg_pool2d(input, kernel_size = (2,1), stride = (2,1), padding = (0,0), ceil_mode=False, count_include_pad=True)
            return y
        def op_calc_1(input):
            y = torch.ops.aten.avg_pool2d(input, kernel_size = (2,1), stride = (2,1), padding = (0,0), ceil_mode=False, count_include_pad=True)
            return y + y
        input = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input)
```
