# aten.add.str

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.add.str(str a, str b) -> str
```
### 1.2 算子功能
`aten.add.str` 是 `aten.add` 的字符串重载（str overload），用于将两个字符串 `a`、`b` 进行拼接（`a + b`），返回拼接后的字符串。


## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `a`           | `str`             | 参与拼接的第一个字符串 |
| `b`           | `str`             | 参与拼接的第二个字符串 |
| 返回值      | `str`| 拼接后的字符串 `a + b` |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | -    |  -     |  -    |  -    | -     | -      | -      | -     | -      | -      | -      |
| NPU    | -    | -    |  -     |  -    |  -    | -     | -      | -      | -     | -      | -      | -      |

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]（计算结果为字符串标量，测试中用其长度包装成 `int32` 张量参与后续计算）


### 2.3 特殊限制说明

> 非张量算子：参数 `a`、`b` 必须为字符串字面量，传入 Tensor 会报 "expected a value of type 'str' for argument 'a'"。标量输入直接用 Python 常量传入（不能用包装张量 `.item()` 取值，否则导出时因数据依赖操作产生 graph break，导致 output_code.py 无法生成），计算结果用张量包装保证能生成计算图。


### 2.4 使用方法


```python
    dtypes = ['int32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    def test_aten_add_str(self, shape):
        def op_calc(tensor, tensor2):
            y = torch.ops.aten.add.str("aten", "_add_str")
            t = torch.full(shape, len(y), device=tensor.device, dtype=torch.int32)
            # 计算结果 + 全 0 张量
            return t + torch.zeros_like(t)

        def op_calc_1(tensor, tensor2):
            y = torch.ops.aten.add.str("aten", "_add_str")
            # 返回原来的计算结果
            return torch.full(shape, len(y), device=tensor.device, dtype=torch.int32)

        tensor = self._generate_tensor(shape, 'int32')
        tensor2 = torch.full(shape, 2, dtype=torch.int32, device="npu")
        self.optype = 'VV'
        self.execute_inductor_test(shape, 'int32', op_calc, op_calc_1, tensor, tensor2)
```
