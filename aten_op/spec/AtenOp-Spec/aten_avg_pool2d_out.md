# aten.avg_pool2d.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.avg_pool2d.out(
    self: Tensor,
    kernel_size: int[2],
    stride: int[2] = [],
    padding: int[2] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int? = None,
    *,
    out: Tensor
) -> Tensor
```
### 1.2 算子功能
`aten.avg_pool2d.out` 是 `aten.avg_pool2d` 的 out 变体（out variant），其功能与 `aten.avg_pool2d` 一致：对输入张量 `self` 执行 2D 平均池化，并将计算结果直接覆盖写入到预先指定的 `out` 张量中。

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
| `out` | `Tensor` | 输出张量，其数据类型（Dtype）和设备（Device）必须与输入张量匹配，形状必须与预期输出严格一致 |
| 返回值 | `Tensor` | `out` 张量本身 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × |
| NPU | - | × | - | - | - | × | - | × | √ | √ | √ | × |

> 注：社区不支持 `int8`、`int32` 和 `int64` 类型，当前 CPU 和 NPU 的 eager 模式同样不支持这些类型。
#### 2.2.2 Shape 支持

1、shape维度范围：[3-4]（3D `(C, H, W)` 或 4D `(N, C, H, W)`，3D 时按单样本 N=1 处理）

2、`H`、`W` 需大于等于 `kernel_size`，`out` 的形状必须与池化后的输出形状一致

### 2.3 特殊限制说明

> `kernel_size`、`stride`、`padding` 为长度为 2 的列表，分别对应 `H`、`W` 两个维度。使用 ChannelsLast 内存格式时仅支持 4D 输入。`out` 张量的形状必须与池化输出形状严格一致，数据类型和设备必须与输入匹配。

### 2.4 使用方法

```python
    dtypes = ['float16', 'bfloat16', 'float32', 'int8', 'int32', 'int64']
    shapes = [(128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_avg_pool2d(self, shape, dtype):
        def op_calc(input, output):
            y = torch.ops.aten.avg_pool2d.out(input, kernel_size = (2,1), stride = (2,1), padding = (0,0), ceil_mode=False, count_include_pad=True, out = output)
            return y
        def op_calc_1(input, output):
            y = torch.ops.aten.avg_pool2d.out(input, kernel_size = (2,1), stride = (2,1), padding = (0,0), ceil_mode=False, count_include_pad=True, out = output)
            return y + y
        input = self._generate_tensor(shape, dtype)
        if input.dim() == 4:
            # (N, C, H, W) -> (N, C, H//2, W)
            out_shape = (input.shape[0], input.shape[1],
                        input.shape[2] // 2, input.shape[3])
        elif input.dim() == 3:
            # (C, H, W) -> (C, H//2, W)
            out_shape = (input.shape[0], input.shape[1] // 2,
                        input.shape[2])
        else:
            raise ValueError(f"Expected 3D or 4D input, got {input.dim()}D")

        output = torch.empty(out_shape, dtype=input.dtype, device='npu')
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, input, output)
```

