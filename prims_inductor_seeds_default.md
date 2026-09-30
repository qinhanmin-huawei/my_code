# prims.inductor_seeds.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.prims.inductor_seeds.default(int count, Device device) -> Tensor
```

### 1.2 算子功能

批量生成随机种子：`y.shape = (count,)`，供多个随机生成操作使用。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `count` | `int` | 种子个数 |
| `device` | `Device` | 生成种子的设备 |
| 返回值 | `Tensor` | int64 种子向量 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | 返回值 | - | - | - | - | - | - | - | √ | - | - | - | - |
| NPU | 返回值 | - | - | - | - | - | - | - | √ | - | - | - | - |

输出 dtype 固定为 int64。

#### 2.2.2 Shape 支持

输出形状为 `(count,)`；`count >= 0`。

### 2.3 特殊限制说明

> 没有 dtype 参数；非空调用会消耗随机状态，`count=0` 时返回空向量。须先导入 `torch._inductor.inductor_prims` 完成注册。

### 2.4 使用方法

不要求生成 kernel，使用自定义回调校验，空用例不计时。

```python
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    counts = list(dict.fromkeys([0, 1] + [shape[-1] for shape in shapes]))

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

    @parametrize('count', counts)
    def test_prims_inductor_seeds_default(self, count):
        import torch._inductor.inductor_prims

        def op_calc(count, device):
            return torch.ops.prims.inductor_seeds.default(count, device)

        device = self._generate_tensor((), 'int64').device
        devices = [] if device.type == 'cpu' else [device.index]
        validator = partial(self._validate_seeds, shape=(count,), device=device, check_rng=count == max(self.counts))
        with torch.random.fork_rng(devices=devices, device_type=device.type):
            self.execute_inductor_test((count,), 'int64', op_calc, op_calc, count, device, test_mode='non_kernel', cpu_reference=False, benchmark=None if count == 0 else 'profiler', result_validator=validator)
```
