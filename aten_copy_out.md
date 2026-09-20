# aten.copy.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.copy.out(Tensor self, Tensor src, bool non_blocking=False, *, Tensor(a!) out) -> Tensor(a!)
```
### 1.2 算子功能
`aten.copy.out` 是 `aten.copy` 的 out 变体，将源张量`src`中的数据拷贝到目标张量`self`中，并将结果写入预分配的输出张量`out`。若`src`与`self`的 dtype 不同，会按需进行类型转换后再拷贝。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`        | `Tensor`          | 目标张量，`src`的数据会被拷贝到该张量中 |
| `src`         | `Tensor`          | 源数据张量 |
| `non_blocking`| `bool`            | 是否执行异步拷贝，默认 `False` |
| `out`         | `Tensor(a!)`      | 输出张量 |
| 返回值      | `Tensor(a!)`| 即 `out` |

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> 无

### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_copy_out(self, shape, dtype):
        def op_calc(tensor, tensor2):
            out = torch.empty_like(tensor)
            torch.ops.aten.copy.out(tensor, tensor2, out=out)
            return out + out

        def op_calc_1(tensor, tensor2):
            out = torch.empty_like(tensor)
            torch.ops.aten.copy.out(tensor, tensor2, out=out)
            return out

        tensor = self._generate_tensor(shape, dtype)
        tensor2 = torch.full(shape, 2, dtype=getattr(torch, dtype), device="npu")
        self.optype = 'VV'
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, tensor2)
```
