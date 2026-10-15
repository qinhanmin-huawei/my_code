# aten.lift.out

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.lift.out(self: Tensor, *, out: Tensor) -> Tensor
```
### 1.2 算子功能

1、`aten.lift.out` 是恒等算子 `lift` 的 `out` 重载：把 `self` 的内容原样写入调用方提供的 `out`，并返回 `out` 本身。

2、eager 下不分配新张量；该 overload 由 codegen 从 `lift` 自动生成（autogen），没有独立的 dispatcher kernel。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 参与计算的源张量 |
| `out` | `Tensor(a!)` | 调用方预分配的输出张量，形状与 dtype 须与 `self` 一致 |
| 返回值 | `Tensor(a!)` | 即 `out` 自身，与入参 `out` 是同一个张量 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |
| NPU    | -    | √    |  -     |  -    |  -    | √     | -      | √      | √     | √      | √      | √      |

与 `aten.lift.default` 相同，不参与数值计算、不做类型提升，7 种基础类型均可用，`out.dtype` 恒等于 `self.dtype`。

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、输出形状与 `self` 完全一致，无广播、无形状变换；`out` 的形状须与 `self` 一致。


### 2.3 特殊限制说明

1、`lift` 是 no-op 算子（ATen 实现即 `at::Tensor lift(const at::Tensor& self) { return self; }`），inductor 把它与 `alias` / `detach` / `prims.view_of` 一起当 nop 消掉，**任何后端都不会生成自己的 kernel**。

2、该 overload 在 Dynamo 侧结构性不可编译，**与 dtype、shape、后端都无关**。原因是 schema 的 `out` 带别名标注 `Tensor(a!)`，而 `.out` 版本由 codegen 自动生成、既没有独立 dispatcher kernel 也没有 functionalization 规则，Dynamo 在 AOTAutograd 之前就拒绝，实测报：

```
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Found a custom (non-ATen) operator whose output has alias annotations:
aten::lift.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!). We only support functionalizing
operators whose outputs do not have alias annotations (e.g. 'Tensor(a)' is a Tensor with an
alias annotation whereas 'Tensor' is a Tensor without. ...)
```

3、eager 下调用正常，返回的就是入参 `out` 本身（`ret is out` 为 True）；限制只出现在 `torch.compile` 路径上。


### 2.4 使用方法

暂无可用测试用例。eager 语义虽然正常，但 2.3 第 2 条的 Dynamo 拒绝发生在图捕获阶段，本仓 `execute_inductor_test` 走 `torch.compile`，用例会在编译期直接抛 `BackendCompilerFailed`。
