# aten.clone.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.clone.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
```
### 1.2 算子功能
`aten.clone.out` 是 `aten.clone` 的 out 变体，将 `self` 的拷贝写入 `out`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 待拷贝的源张量 |
| `memory_format` | `MemoryFormat?` | 可选内存格式，默认 None |
| `out` | `Tensor(a!)` | 输出张量，需与 `self` 同形状同 dtype |
| 返回值 | `Tensor(a!)` | 即 `out` |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| 设备 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |
| NPU | - | √ | - | - | - | √ | - | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

### 2.3 特殊限制说明

> `out` 需预分配为与 `self` 同形状、同 dtype 的张量。

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]
    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_clone_out(self, shape, dtype):
        def op_calc(tensor, tensor2):
            out = torch.empty_like(tensor)
            torch.ops.aten.clone.out(tensor, out=out)
            return out + out
        def op_calc_1(tensor, tensor2):
            out = torch.empty_like(tensor)
            torch.ops.aten.clone.out(tensor, out=out)
            return out
        tensor = self._generate_tensor(shape, dtype)
        tensor2 = torch.full(shape, 2, dtype=getattr(torch, dtype), device="npu")
        self.optype = 'VV'
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, tensor2)
```
