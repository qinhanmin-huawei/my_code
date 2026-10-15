# flex_attention_backward

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.higher_order.flex_attention_backward(query, key, value, out, logsumexp, grad_out, grad_logsumexp, fw_graph, joint_graph, block_mask, scale, kernel_options, score_mod_other_buffers=(), mask_mod_other_buffers=()) -> tuple
```

### 1.2 算子功能

计算带可编程分数变换的注意力对 query、key、value 的梯度。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `query / key / value` | `Tensor` | 同 dtype 的注意力输入。 |
| `score_mod` | `Callable` | 恒等分数变换。 |
| `block_mask` | `tuple` | 无稀疏屏蔽的 block mask。 |
| `scale` | `float` | head_dim 的平方根倒数。 |
| `kernel_options` | `dict` | 内核配置。 |
| `out / logsumexp` | `Tensor` | 前向结果与以 2 为底的 logsumexp。 |
| `grad_out / grad_logsumexp` | `Tensor` | 输出梯度。 |
| `joint_graph` | `Callable` | 分数变换的梯度。 |
| `返回值` | `tuple` | 注意力结果或输入梯度。 |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | x | - | - | - | x | - | x | x | x | x | - |
| NPU | - | x | - | - | - | x | - | x | x | x | x | - |

#### 2.2.2 Shape 支持

`query`、`key` 和 `value` 固定为 4 维 Tensor。

### 2.3 特殊限制说明

报错为：eager failure: vmap(joint_graph, ..., out_dims=[0, None, None, None, None])(<inputs>): out_dims is not compatible with the structure of `outputs`。与社区报错一致，社区在2.14版本中已修复，pr号：#188869。在2.10版本暂不支持。

### 2.4 使用方法

```python
    shapes = [(4, 8, 16, 32)]
    dtypes = ['float16', 'bfloat16', 'float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_flex_attention_backward(self, shape, dtype):
        def op_calc(query, key, value, score_mod, block_mask, scale, options, joint_graph, grad_out, grad_logsumexp):
            forward = torch.ops.higher_order.flex_attention(query, key, value, score_mod, block_mask, scale, options)
            y = torch.ops.higher_order.flex_attention_backward(query, key, value, forward[0], forward[1], grad_out, grad_logsumexp, score_mod, joint_graph, block_mask, scale, options)
            return (y[0], y[1], y[2])

        def op_calc_1(query, key, value, score_mod, block_mask, scale, options, joint_graph, grad_out, grad_logsumexp):
            forward = torch.ops.higher_order.flex_attention(query, key, value, score_mod, block_mask, scale, options)
            y = torch.ops.higher_order.flex_attention_backward(query, key, value, forward[0], forward[1], grad_out, grad_logsumexp, score_mod, joint_graph, block_mask, scale, options)
            y = (y[0], y[1], y[2])
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

        def joint_graph(score, batch, head, query_index, key_index, grad):
            return grad, None, None, None, None

        grad_out = self._generate_tensor(shape, dtype, min=-2, max=2)
        grad_logsumexp = self._generate_tensor(shape[:-1], 'float32', min=-2, max=2)
        self.execute_inductor_test(
            shape, dtype, op_calc, op_calc_1, query, key, value, score_mod, block_mask, scale, options, joint_graph, grad_out, grad_logsumexp,
            op_name='torch.ops.higher_order.flex_attention_backward',
        )
```
