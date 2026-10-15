# prims._make_token.default

## 1. OP 概述

### 1.1 算子原型

```text
torch.ops.prims._make_token.default() -> Tensor
```

### 1.2 算子功能

创建用于跟踪副作用操作执行顺序的初始 token，配合 `with_effects` 等机制表达操作间的依赖关系；token 本身不承载数值，也不是设备同步指令。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| 返回值 | `Tensor` | 表示控制依赖的空 token |

### 2.2 支持规格

#### 2.2.1 DataType 支持

不适用：该算子无数值输入，也无 dtype 参数；辅助 Tensor 的 dtype 不是该算子的输入 dtype。

#### 2.2.2 Shape 支持

token 的形状为 `(0,)`，不承载元素值。

### 2.3 特殊限制说明

- 本地服务器已将 `_sink_tokens` 移出 fallback list，`_make_token` → `_sink_tokens` 路径验证通过；测试服务器未将 `_sink_tokens` 移出 fallback list，该路径验证失败。
- 测试服务器采用 2.4 中的测试脚本，未使用 `_sink_tokens` 消费 token，仅自动确认 `_make_token` 被 Dynamo 捕获；本次 `output_code` 未保留 token，仅包含辅助加法，不验证消费链路及副作用顺序。

### 2.4 使用方法

不要求生成 kernel，关闭性能计时；数值校验仅覆盖辅助 Tensor 的加法结果。

```python
import torch
from torch._dynamo.backends.registry import lookup_backend
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestPrimsMakeToken(TestUtils):
    shapes = TestUtils._pointwise_demo_shapes[:1]
    dtypes = ['float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_prims_make_token_default(self, shape, dtype):
        def op_calc(tensor):
            token = torch.ops.prims._make_token.default()
            return tensor + tensor

        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc, tensor, test_mode='non_kernel', benchmark=None)
        graph_checked = False
        inductor_backend = lookup_backend('inductor')

        def checked_backend(gm, example_inputs):
            nonlocal graph_checked
            self.assertTrue(
                any(node.op == 'call_function' and node.target == torch.ops.prims._make_token.default
                    for node in gm.graph.nodes),
                'make_token was not captured in the Dynamo graph',
            )
            graph_checked = True
            return inductor_backend(gm, example_inputs)

        try:
            compiled = torch.compile(op_calc, backend=checked_backend, fullgraph=True)
            compiled(tensor)
            self.assertTrue(graph_checked, 'make_token graph check did not run')
        except Exception:
            self.error_message = 'inductor failure'
            raise
        print('make_token captured in Dynamo graph; no consumer or side-effect ordering is tested.')


instantiate_parametrized_tests(TestPrimsMakeToken)


if __name__ == "__main__":
    run_tests()
```
