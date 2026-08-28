# aten.index.Tensor_hacked_twin

## 1. OP 概述
### 1.1 算子原型
```
 torch.ops.aten.index.Tensor_hacked_twin(self: Tensor, indices: List[Optional[Tensor]]) -> Tensor
```
### 1.2 算子功能
`aten.index.Tensor_hacked_twin`：根据指定的张量列表（List[Tensor]），对源张量进行多维非连续的非轴对齐索引，并提取相应位置的数据。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self/input`        | `Tensor`          | 被索引的源张量 |
| `indices`        | `List[Optional[Tensor]]`| 索引张量列表|
| 返回值      | `Tensor`| 索引后产生的新张量 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | ×      |


#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]


### 2.3 特殊限制说明

> indices列表中的每一个 Tensor 对应 self 的一个维度。其元素必须是整数类型或布尔类型的张量。对于不需要索引的轴，通常填入 None 进行对齐。列表的长度不能超过 self 的总维数。


### 2.4 使用方法


```python
    dtypes = ['int8', 'int32', 'int64', 'float16', 'bfloat16', 'float32']
    shapes = [(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]

    def test_aten_index_Tensor_hacked_twin(self, shape, dtype):
        tensor = self._generate_tensor(shape, dtype)
        max_index = len(shape)
        index_size = min(5, max_index)
        index_tensor = torch.randint(0, max_index, size=(index_size,), dtype=torch.long).to('npu')

        def op_calc(tensor, index_tensor):
            result = torch.ops.aten.index.Tensor_hacked_twin(tensor, index_tensor) + 1
            return result
        def op_calc_1(tensor, index_tensor):
            result = torch.ops.aten.index.Tensor_hacked_twin(tensor, index_tensor) + 1
            return result + result
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, tensor, [index_tensor])
```
