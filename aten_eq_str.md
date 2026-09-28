# aten.eq.str

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.eq.str(str a, str b) -> bool
```
### 1.2 算子功能
`aten.eq.str` 是 `aten.eq` 的字符串重载（str overload），用于比较两个字符串 `a`、`b` 是否相等（`a == b`），返回 Python `bool`。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `a`           | `str`             | 参与比较的第一个字符串 |
| `b`           | `str`             | 参与比较的第二个字符串 |
| 返回值      | `bool`| 相等比较结果 `a == b` |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | -    |  -     |  -    |  -    | -     | -      | -      | -     | -      | -      | -      |
| NPU    | -    | -    |  -     |  -    |  -    | -     | -      | -      | -     | -      | -      | -      |

#### 2.2.2 Shape 支持

不适用（无张量输入）。


### 2.3 特殊限制说明

> 非张量算子：参数 `a`、`b` 必须为字符串字面量，传入 Tensor 会报 "expected a value of type 'str' for argument 'a'"。比较结果为 Python `bool` 标量，测试中通过与全零 Tensor 相加承接结果，保证能生成计算图。


### 2.4 使用方法


```python
    def test_aten_eq_str(self):
        def op_calc(anchor, first, second):
            return anchor + torch.ops.aten.eq.str(first, second)

        def op_calc_1(anchor, first, second):
            y = torch.ops.aten.eq.str(first, second)
            return anchor + y + y

        anchor = torch.zeros((128,), dtype=torch.float32, device='npu')
        self.execute_inductor_test((128,), 'float32', op_calc, op_calc_1,
                                   anchor, 'aten', 'aten', op_name='aten.add')
```
