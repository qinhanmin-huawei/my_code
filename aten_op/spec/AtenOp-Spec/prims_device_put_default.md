# prims.device_put.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.prims.device_put.default(
    Tensor a, Device device, bool non_blocking=False
) -> Tensor
```

### 1.2 算子功能

将张量放到指定设备：`y = a.to(device)`，保持元素值、shape 和 dtype。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `Tensor` | 源张量 |
| `device` | `Device` | 目标设备 |
| `non_blocking` | `bool` | 是否允许异步传输，默认 False |
| 返回值 | `Tensor` | 位于目标设备的张量 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | a | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | a | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

输入和输出的目标维度范围：[1-4]，输出 shape 与输入一致。

### 2.3 特殊限制说明

> 源设备和目标设备须可用；当前规格采用 CPU 与 GPU/NPU 间的同步迁移。同设备调用不保证发生实际数据搬运。

### 2.4 使用方法

不要求生成 kernel，采用 `end_to_end` 端到端计时。

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    cases = [
        dict(shape=shape, direction=direction)
        for shape in shapes
        for direction in ('from_cpu', 'to_cpu')
    ]
    cases.extend(dict(shape=shape, direction='same_device') for shape in shapes[:1])

    @parametrize('case', cases)
    @parametrize('dtype', dtypes)
    def test_prims_device_put_default(self, case, dtype):
        shape = case['shape']
        direction = case['direction']

        def op_calc(tensor, device):
            y = torch.ops.prims.device_put.default(tensor, device, non_blocking=False)
            return y

        def op_calc_1(tensor, device):
            y = torch.ops.prims.device_put.default(tensor, device, non_blocking=False)
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        sync_device = tensor.device
        device = torch.device('cpu') if direction == 'to_cpu' else tensor.device
        if direction == 'from_cpu':
            tensor = tensor.cpu()

        def synchronize():
            torch.npu.synchronize(sync_device)

        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, device, test_mode='non_kernel', cpu_reference=False, benchmark='end_to_end', synchronize=synchronize)
```
