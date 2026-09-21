# aten.lift.default

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.lift.default(self: Tensor) -> Tensor
```
### 1.2 算子功能

1、`aten.lift.default` 是恒等算子，原样返回 `self`，不做任何数值运算、不复制数据。

2、返回的张量与 `self` 共享底层存储，是纯别名（View）语义；它不是公开 Python API，只能经 `torch.ops.aten.lift.default` 调用。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 参与计算的源张量 |
| 返回值 | `Tensor` | 与 `self` 共享存储、形状与元素完全一致的张量 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

`lift` 不参与数值计算、也不做类型提升，实现就是 `return self`，因此 dtype 支持等价于「张量本身能否存在」，上表 7 种基础类型均可用，返回 dtype 恒等于 `self.dtype`。

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、输出形状与 `self` 完全一致，无广播、无形状变换。


### 2.3 特殊限制说明

1、`lift` 是 no-op 算子。ATen 实现原文即 `// The default implementation of lift is a no-op.` → `at::Tensor lift(const at::Tensor& self) { return self; }`（`TensorShape.cpp`），dispatch 为 `CompositeExplicitAutograd`；inductor 把它与 `alias` / `detach` / `detach_` / `prims.view_of` 归成一类直接消掉（`@register_lowering([aten.alias, aten.detach, aten.detach_, aten.lift, prims.view_of]) def nop(x): return x`）。**该算子在任何后端都不会生成自己的 kernel**，NPU 侧 op-plugin 也没有为它注册内核——不需要，因为实现就是 `return self`。

2、由第 1 条直接导致：`lift` 单独成图时 inductor 产出 0 个 `output_code`，编译期校验报

```
[failure]no output_code generated
```

这不是算子报错，而是「no-op 没有 kernel 可查」的必然结果。要让图里有 kernel 可查，必须在 `lift` 之外再挂一个非 no-op 运算（如 `* 2`）托底；但那样验证到的是载体运算，**不能据此宣称 `lift` 本身被 lowering 了**。

3、`torch.lift` 不是公开 Python API（`hasattr(torch, 'lift')` 为 False），入口只有 `torch.ops.aten.lift.default` / `.out`。


### 2.4 使用方法

暂无可用测试用例。原因是 2.3 第 1 条的 no-op 语义：本仓 `execute_inductor_test` 要求图内产出 device kernel，而 `lift` 自身不产出任何 `output_code`，用例会被判 `[failure]no output_code generated`；挂载体运算托底又会把被测对象变成载体本身，故本 overload 不出测试用例。
