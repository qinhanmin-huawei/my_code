# aten.neg.int

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.neg.int(int a) -> int
```

### 1.2 算子功能
`aten.neg.int` 对整数标量执行取负，返回整数标量结果。

## 2. OP 规格
### 2.1 参数说明
| 参数名 | 类型 | 说明 |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `a` | `int` | 待取负的整数标量 |
| 返回值 | `int` | 输入整数标量的相反数 |

### 2.2 支持规格
#### 2.2.1 DataType 支持
|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool | complex64 | complex128 |
| ------ | ---- | ---- |  ----  |  ---- |  ----  | ----- | ----- | ----- | ----- | ------ | ------ | ------ | --------- | ---------- |
| GPU    | -    | √    | -      | -     | -      | √     | -     | √     | ×     | ×      | ×      | ×      | ×         | ×          |
| NPU    | -    | √    | -      | -     | -      | √     | -     | √     | ×     | ×      | ×      | ×      | -         | -          |

#### 2.2.2 Shape 支持
1、仅支持零维整数标量输入，不涉及 shape 泛化。

2、输入参数 `a` 支持 `int8`、`int32`、`int64`；复数类型不属于该重载的输入类型。

### 2.3 特殊限制说明
> 标量结果可能在编译阶段常量折叠；用例通过零值 Tensor 承接结果验证 eager/Inductor 语义，允许 output code 中不保留 `neg.int`。

### 2.4 使用方法
```python
    def test_aten_neg_int(self):
        def op_calc(anchor, scalar):
            scalar_result = torch.ops.aten.neg.int(scalar)
            return anchor + scalar_result

        def op_calc_1(anchor, scalar):
            scalar_result = torch.ops.aten.neg.int(scalar)
            return anchor + scalar_result + scalar_result

        anchor = torch.zeros((128,), dtype=torch.int32, device='npu')
        self.execute_inductor_test((128,), 'int32', op_calc, op_calc_1,
                                   anchor, 3, op_name='aten.add')
```