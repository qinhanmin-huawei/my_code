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

> 无输入 dtype 参数；token 用于依赖传递，不应当作数值张量参与精度或设备 kernel 性能比较。

> 当前 NPU 验证使用已将 `_sink_tokens` 移出强制 fallback 列表的环境；该结果不代表未修改的后端配置也能通过。

### 2.4 使用方法

通过 `_sink_tokens` 消费创建的 token，检查创建—消费链路的编译执行兼容性，不验证真实副作用操作的执行顺序。

不要求生成 kernel，关闭性能计时；自动数值校验仅覆盖辅助 Tensor 的加法结果，不对 token 做数值校验。

token 是否入图需结合 Dynamo/AOT 图确认，当前用例未自动断言图结构；仅凭加法结果正确或 output code 中没有 token，不能判断 token 是否入图。

```python
    shapes = TestUtils._pointwise_demo_shapes[:1]
    dtypes = ['float32']

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_prims_make_token_default(self, shape, dtype):
        def op_calc(tensor):
            token = torch.ops.prims._make_token.default()
            torch.ops.prims._sink_tokens.default([token])
            y = tensor + tensor
            return y

        def op_calc_1(tensor):
            token = torch.ops.prims._make_token.default()
            torch.ops.prims._sink_tokens.default([token])
            y = tensor + tensor
            return y + y

        tensor = self._generate_tensor(shape, dtype)
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, test_mode='non_kernel', benchmark=None)
```
