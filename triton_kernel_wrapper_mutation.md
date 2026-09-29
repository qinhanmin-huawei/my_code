# triton_kernel_wrapper_mutation

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.higher_order.triton_kernel_wrapper_mutation(
    kernel_idx: int, constant_args_idx: int, grid: list,
    tma_descriptor_metadata: dict, kwargs: dict
) -> None
```

此处为图内调用原型；通常由编译器捕获 Triton kernel 启动后生成，不手工设置侧表索引。

### 1.2 算子功能

调用已注册的 Triton kernel，并将其写入的张量变化保留在调用参数中。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `kernel_idx` | `int` | kernel 在侧表中的索引 |
| `constant_args_idx` | `int` | 静态参数在侧表中的索引 |
| `grid` | `list` | kernel 的启动网格 |
| `tma_descriptor_metadata` | `dict` | TMA 描述符信息，不使用时为空 |
| `kwargs` | `dict` | kernel 的运行时参数，包含被写入的 Tensor |
| 返回值 | `None` | 结果通过 Tensor 参数写回 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | kernel 中的输入及写回 Tensor | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | kernel 中的输入及写回 Tensor | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

上表对应示例拷贝 kernel；其他 kernel 的 dtype 支持由其实现决定。

#### 2.2.2 Shape 支持

示例输入和写回张量支持 [1-4] 维，两者 shape、dtype、device 相同且连续；支持元素数非整块的尾部掩码处理。

### 2.3 特殊限制说明

> 需要目标设备的 Triton 运行环境；可写参数、网格和越界掩码必须符合实际 kernel 的约定。

### 2.4 使用方法

先定义拷贝 kernel；下方调用在编译图中生成 `triton_kernel_wrapper_mutation`，以写回张量作为框架比较结果。

```python
import triton
import triton.language as tl


@triton.jit
def _copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0)
    tl.store(output_ptr + offsets, values, mask=offsets < n_elements)
```

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    cases = [dict(shape=shape) for shape in shapes]
    for shape in [shape for shape in shapes if len(shape) == 2][:1]:
        cases.append(dict(shape=(shape[0], max(1, shape[1] - 1))))

    @parametrize('case', cases)
    @parametrize('dtype', dtypes)
    def test_triton_kernel_wrapper_mutation(self, case, dtype):
        shape = case['shape']

        def op_calc(tensor, out):
            _copy_kernel[(triton.cdiv(tensor.numel(), 256),)](tensor, out, tensor.numel(), BLOCK_SIZE=256)
            y = out
            return y

        def op_calc_1(tensor, out):
            _copy_kernel[(triton.cdiv(tensor.numel(), 256),)](tensor, out, tensor.numel(), BLOCK_SIZE=256)
            y = out
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        out = torch.empty_like(tensor)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, out, op_name='torch.ops.higher_order.triton_kernel_wrapper_mutation')
```
