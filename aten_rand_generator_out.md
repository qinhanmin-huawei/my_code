# aten.rand.generator_out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.rand.generator_out(
    SymInt[] size,
    *,
    Generator? generator,
    Tensor(a!) out
) -> Tensor(a!)
```

### 1.2 算子功能
在 `out` Tensor 中生成服从均匀分布 `U[0, 1)` 的随机数，并返回同一个 `out` Tensor。输出 shape 由 `size` 指定；随机数由可选的 `generator` 控制。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `size` | `SymInt[]` | 输出 Tensor 的目标 shape，长度为输出维度数 |
| `generator` | `Generator?` | 可选随机数生成器；必须与 `out` 的设备兼容 |
| `out` | `Tensor(a!)` | 承接随机数并被原地写入的输出 Tensor |
| 返回值 | `Tensor(a!)` | 返回与 `out` 具有相同别名的 Tensor |

### 2.2 支持规格
#### 2.2.1 DataType 支持
| | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | - | - |
| NPU | - | × | - | - | - | × | - | × | √ | √ | √ | × | - | - |

输出 dtype 由 `out.dtype` 决定；本规格只考察浮点输出，复数及非浮点 dtype 不纳入 NPU 考察。

#### 2.2.2 Shape 支持
本规格覆盖 1-4 维输出 Tensor。`out.shape` 必须与 `tuple(size)` 一致，输出元素数量为 `prod(size)`。

### 2.3 特殊限制说明
> `generator` 和 `out` 必须通过关键字参数传入，并且其 device、dtype、shape 应与 `size` 兼容。传入 `generator=None` 时使用当前设备的默认随机数生成器。随机结果依赖 Generator 状态；用例使用固定 seed 的 Generator，并依赖测试框架分别深拷贝 eager 与 Inductor 参数，使两条路径从相同 Generator 状态开始。该算子写入 `out`，返回值与 `out` 具有别名关系。

### 2.4 使用方法
```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_rand_generator_out(self, shape, dtype):
        def op_calc(size, generator, out_tensor):
            return torch.ops.aten.rand.generator_out(
                size, generator=generator, out=out_tensor)

        def op_calc_1(size, generator, out_tensor):
            y = torch.ops.aten.rand.generator_out(
                size, generator=generator, out=out_tensor)
            return y + y

        generator = torch.Generator(device='npu')
        generator.manual_seed(0)
        out_tensor = torch.empty(shape,
                                 dtype=eval('torch.' + dtype),
                                 device='npu')
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1,
                                   tuple(shape), generator, out_tensor)
```