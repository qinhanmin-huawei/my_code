# flex_attention

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.higher_order.flex_attention(query, key, value, score_mod, block_mask, scale, kernel_options, score_mod_other_buffers=(), mask_mod_other_buffers=()) -> tuple
```

### 1.2 算子功能

计算带可编程分数变换的注意力输出及 logsumexp。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `query / key / value` | `Tensor` | 同 dtype 的注意力输入。 |
| `score_mod` | `Callable` | 恒等分数变换。 |
| `block_mask` | `tuple` | 无稀疏屏蔽的 block mask。 |
| `scale` | `float` | head_dim 的平方根倒数。 |
| `kernel_options` | `dict` | 内核配置。 |
| `返回值` | `tuple` | 注意力结果或输入梯度。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |
| NPU | - | x | - | - | - | x | - | x | √ | √ | √ | - |

#### 2.2.2 Shape 支持

1. 本测试覆盖的 shape 维度固定为 `4`。
2. 测试 shape：`(4, 8, 16, 32)`。

### 2.3 特殊限制说明

bf16与f16输入会有精度错误，与gpu一致（相同输入下错误个数一致，最大绝对误差，相对误差均一致）。flex_attention在2.7.1和2.13专项都搞过，错误为2.10版本导致。

### 2.4 使用方法

```python
    shapes = [(4, 8, 16, 32)]
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_flex_attention(self, shape, dtype):
        def op_calc(query, key, value, score_mod, block_mask, scale, options):
            y = torch.ops.higher_order.flex_attention(query, key, value, score_mod, block_mask, scale, options)
            return (y[0], y[1])

        def op_calc_1(query, key, value, score_mod, block_mask, scale, options):
            y = torch.ops.higher_order.flex_attention(query, key, value, score_mod, block_mask, scale, options)
            y = (y[0], y[1])
            return (y[0] + y[0], *y[1:])

        from torch.nn.attention.flex_attention import _create_empty_block_mask

        query = self._generate_tensor(shape, dtype, min=-2, max=2)
        key = self._generate_tensor(shape, dtype, min=-2, max=2)
        value = self._generate_tensor(shape, dtype, min=-2, max=2)
        block_mask = _create_empty_block_mask(query, key).as_tuple()
        scale = shape[-1] ** -0.5

        def score_mod(score, batch, head, query_index, key_index):
            return score

        options = {
            'BACKEND': 'AUTO', 'PRESCALE_QK': False,
            'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False,
            'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False,
        }
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, query, key, value, score_mod, block_mask, scale, options,
            op_name='torch.ops.higher_order.flex_attention',
        )
```
