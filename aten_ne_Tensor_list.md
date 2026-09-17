# aten.ne.Tensor_list

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.ne.Tensor_list(Tensor[] a, Tensor[] b) -> bool
```

### 1.2 算子功能
比较两个 Tensor 列表是否不相等，返回单个 Python `bool`。列表元素应为 Tensor，并按列表顺序和 Tensor 内容进行比较。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `a` | `Tensor[]` | 第一个 Tensor 列表 |
| `b` | `Tensor[]` | 第二个 Tensor 列表 |
| 返回值 | `bool` | 两个列表不相等时为 `True` |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | × | × | × | × | - | - |
| NPU | - | × | - | - | - | × | - | × | × | × | × | × | - | - |

上表针对常规多元素 Tensor 列表输入。返回值是 Python `bool`，不是 bool Tensor。

#### 2.2.2 Shape 支持
输入为 Tensor 列表，列表长度可以不同。经 CPU、GPU 和 NPU 环境实测，对应位置使用多元素 Tensor 时均不能得到单个布尔结果，会报 `Boolean value of Tensor with more than one value is ambiguous`。

常规 1-4 维、多元素 Tensor 列表输入不支持。零维或单元素 Tensor 虽可规避布尔转换歧义，但不代表该 overload 支持常规 Tensor shape，因此不作为本规格的 Inductor 精度和性能测试方案。

### 2.3 特殊限制说明
> `Tensor[]` / `*_list` overload 主要与 TorchScript 的列表类型系统有关，而 TorchScript 架构已被社区弃用。该 overload 目前仍保留在 schema 中，但使用常规多元素 Tensor 时，CPU、GPU 和 NPU eager 均报 `Boolean value of Tensor with more than one value is ambiguous`。由于 eager 基准无法成立，不继续进行 Inductor 精度和性能验证，也不使用零维 Tensor 或 `anchor` 人为绕过限制。该结果只说明当前测试输入和运行时不可用，不能据此判定其 Inductor fallback 或 lowering 状态。

### 2.4 使用方法
以下用例用于复现并确认 eager 限制，不应提交给 `execute_inductor_test`：
```python
    dtypes = ['int8', 'int32', 'int64', 'float16',
              'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_ne_Tensor_list(self, shape, dtype):
        first = self._generate_tensor(shape, dtype)
        second = self._generate_tensor(shape, dtype)
        with self.assertRaisesRegex(
                RuntimeError,
                'Boolean value of Tensor with more than one value is ambiguous'):
            torch.ops.aten.ne.Tensor_list([first], [second])
```

## 3. 遗留问题
该 `*_list` overload 与 TorchScript 的列表类型系统有关。TorchScript 架构已被社区弃用，此类 overload 主要为历史兼容而保留，部分接口未标记 deprecated，但不代表仍会持续完善。当前该算子使用常规多元素 Tensor 时，CPU、GPU 和 NPU eager 均无法执行，因此不进行 Inductor 精度和性能验证。