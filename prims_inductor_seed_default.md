# prims.inductor_seed.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.prims.inductor_seed.default(Device device) -> Tensor
```

### 1.2 算子功能

在指定设备上生成一个新的随机种子，供编译路径中的随机生成操作使用。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `device` | `Device` | 生成种子的设备 |
| 返回值 | `Tensor` | int64 标量种子 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | 返回值 | - | - | - | - | - | - | - | √ | - | - | - | - |
| NPU | 返回值 | - | - | - | - | - | - | - | √ | - | - | - | - |

输出 dtype 固定为 int64；这不是七种输入 dtype 的数值算子。

#### 2.2.2 Shape 支持

输出形状固定为 `()`。

### 2.3 特殊限制说明

> 没有 dtype 参数；调用会消耗随机状态。须先导入 `torch._inductor.inductor_prims` 完成注册。

### 2.4 使用方法

不要求生成 kernel，使用自定义回调校验。

```python
    def _validate_seeds(self, eager_result, inductor_result, *, eager_fn, compiled_fn, args, shape, device, check_rng):
        for result in (eager_result, inductor_result):
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(tuple(result.shape), shape)
            self.assertEqual(result.dtype, torch.int64)
            self.assertEqual(result.device, device)
        if not check_rng:
            return

        rng = torch if device.type == 'cpu' else getattr(torch, device.type)
        rng_args = () if device.type == 'cpu' else (device,)
        saved_state = rng.get_rng_state(*rng_args)
        try:
            for fn in (eager_fn, compiled_fn):
                rng.set_rng_state(saved_state, *rng_args)
                first = fn(*args).clone()
                state_after = rng.get_rng_state(*rng_args)
                second = fn(*args).clone()
                third = fn(*args).clone()
                self.assertFalse(
                    torch.equal(first, second) and torch.equal(first, third),
                    'three consecutive seed outputs are identical',
                )
                rng.set_rng_state(saved_state, *rng_args)
                repeated = fn(*args)
                torch.testing.assert_close(repeated, first, rtol=0, atol=0)
                self.assertFalse(torch.equal(saved_state, state_after), 'seed generation did not advance RNG state')
        finally:
            rng.set_rng_state(saved_state, *rng_args)

    def test_prims_inductor_seed_default(self):
        import torch._inductor.inductor_prims

        def op_calc(device):
            return torch.ops.prims.inductor_seed.default(device)

        device = self._generate_tensor((), 'int64').device
        devices = [] if device.type == 'cpu' else [device.index]
        validator = partial(self._validate_seeds, shape=(), device=device, check_rng=True)
        with torch.random.fork_rng(devices=devices, device_type=device.type):
            self.execute_inductor_test((), 'int64', op_calc, op_calc, device, test_mode='non_kernel', cpu_reference=False, result_validator=validator)
```
