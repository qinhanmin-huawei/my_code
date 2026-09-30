# prims.inductor_random.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.prims.inductor_random.default(SymInt[] size, Tensor seed, str mode) -> Tensor
```

### 1.2 算子功能

在 seed 所在设备生成随机张量：`mode='rand'` 为 `[0, 1)` 均匀分布，`mode='randn'` 为标准正态分布。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `size` | `SymInt[]` | 输出形状 |
| `seed` | `Tensor` | 编译路径使用的 int64 标量种子 |
| `mode` | `str` | `'rand'` 或 `'randn'` |
| 返回值 | `Tensor` | fp32 随机张量 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | 参数 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | seed | - | - | - | - | - | - | - | √ | - | - | - | - |
| NPU | seed | - | - | - | - | - | - | - | √ | - | - | - | - |
| GPU | 返回值 | - | - | - | - | - | - | - | - | - | √ | - | - |
| NPU | 返回值 | - | - | - | - | - | - | - | - | - | √ | - | - |

没有输出 dtype 参数，输出固定为 fp32。

#### 2.2.2 Shape 支持

`seed` 的形状为 `()`；输出目标维度范围：[1-4]，形状由 size 指定。
各维长度非负；任一维长度为 0 时返回对应 shape 的空张量，不要求生成计算 kernel。

### 2.3 特殊限制说明

> 上游 Eager 参考只使用 `seed.device`，不使用 seed 数值；编译路径使用该种子。两条路径不保证随机样本逐元素一致。

### 2.4 使用方法

所有用例使用自定义回调校验，空用例不要求生成 kernel 且不计时。

```python
    shapes = [(16,), (16, 32), (16, 32, 64), (16, 32, 64, 128)]
    representative_shape = max(shapes, key=math.prod, default=(0,))
    random_shapes = shapes + [(0,)]

    def _check_distribution(self, result, mode, low, high, scale):
        if result.numel() < 4096:
            return
        mean, variance = {
            'rand': (0.5, 1 / 12),
            'randn': (0.0, 1.0),
            'randint': ((low + high - 1) / 2, ((high - low)**2 - 1) / 12),
        }[mode]
        sample = result.detach().reshape(-1)[:65536].to(device='cpu', dtype=torch.float64) / scale
        count = sample.numel()
        mean_tolerance = max(1e-6, 8 * math.sqrt(variance / count))
        variance_tolerance = max(1e-6, 8 * variance * math.sqrt(2 / (count - 1)))
        self.assertLessEqual(abs(sample.mean().item() - mean), mean_tolerance, 'unexpected sample mean')
        self.assertLessEqual(abs(sample.var(unbiased=False).item() - variance), variance_tolerance, 'unexpected sample variance')

    def _validate_random(self, eager_result, inductor_result, *, eager_fn, compiled_fn, args, mode, low=0, high=1):
        seed, shape = args[:2]
        scale = 2 if self.used_func == 'func_plus' else 1
        check_randomness = tuple(shape) == tuple(self.representative_shape)
        for result in (eager_result, inductor_result):
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(tuple(result.shape), tuple(shape))
            self.assertEqual(result.device, seed.device)
            self.assertEqual(result.dtype, torch.int64 if mode == 'randint' else torch.float32)
            if result.numel() == 0:
                continue
            if mode == 'randint':
                self.assertTrue(((result >= low * scale) & (result <= (high - 1) * scale)).all().item(), 'randint result outside scaled bounds')
                self.assertTrue((result.remainder(scale) == 0).all().item(), 'randint result is not a multiple of scale')
            else:
                self.assertTrue(torch.isfinite(result).all().item(), 'random result contains NaN or Inf')
                if mode == 'rand':
                    self.assertTrue(((result >= 0) & (result < scale)).all().item(), 'rand result outside scaled [0, 1)')
            if check_randomness:
                self._check_distribution(result, mode, low, high, scale)

        if check_randomness and inductor_result.numel() != 0:
            expected = inductor_result.clone()
            repeated = compiled_fn(*args)
            torch.testing.assert_close(repeated, expected, rtol=0, atol=0)

    @parametrize('shape', random_shapes)
    @parametrize('mode', ['rand', 'randn'])
    def test_prims_inductor_random_default(self, shape, mode):
        import torch._inductor.inductor_prims

        def op_calc(seed, shape, mode):
            y = torch.ops.prims.inductor_random.default(shape, seed, mode)
            return y

        def op_calc_1(seed, shape, mode):
            y = torch.ops.prims.inductor_random.default(shape, seed, mode)
            return y + y

        seed = self._generate_tensor((), 'int64')
        empty = math.prod(shape) == 0
        devices = [] if seed.device.type == 'cpu' else [seed.device.index]
        validator = partial(self._validate_random, mode=mode)
        with torch.random.fork_rng(devices=devices, device_type=seed.device.type):
            self.execute_inductor_test(shape, 'float32', op_calc, op_calc_1, seed, shape, mode, op_name='torch.ops.prims.inductor_random.default', test_mode='non_kernel' if empty else 'default', cpu_reference=False, benchmark=None if empty else 'profiler', result_validator=validator)
```
