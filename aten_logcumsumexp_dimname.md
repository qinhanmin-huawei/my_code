# aten.logcumsumexp.dimname

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.logcumsumexp.dimname(self: Tensor, dim: Dimname) -> Tensor
```
该原型对应仍保留 Named Tensor 的旧版接口。
### 1.2 算子功能

1、`aten.logcumsumexp.dimname` 是 `logcumsumexp` 的具名维度重载：沿名称为 `dim` 的维度做 `log(sum(exp(x)))` 的累积扫描，语义与 `aten.logcumsumexp.default` 完全一致。

2、与 `int dim` 版本的区别只在于维度用字符串名指定，ATen 侧先做 `dimname_to_position(self, dim)` 换成位置索引再落同一套内核。


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 带维度名称的输入张量 |
| `dim` | `Dimname` | 待扫描的维度名称，Python 中传字符串 |
| 返回值 | `Tensor` | 沿 `dim` 累积 logsumexp 后的张量，dtype 与 `self` 一致 |


### 2.2 支持规格

#### 2.2.1 DataType 支持

|        |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 |  int64 | fp16  | fp32   | bf16   | bool   |
| ------ | ---- | ---- |  ----  |  ---- |  ---- | ----- | -----  |  ----- | ----- | ------ | ------ | ------ |
| GPU    | -    | ×    |  -     |  -    |  -    | ×     | -      | ×      | ×     | ×      | ×      | -      |
| NPU    | -    | ×    |  -     |  -    |  -    | ×     | -      | ×      | ×     | ×      | ×      | -      |

中标 `×` 的六种是本仓用例覆盖的 dtype（int8 / int32 / int64 / fp16 / fp32 / bf16）：具名张量进图会被 inductor 直接拒绝（见 2.3 第 2 条），六种都跑不出结果，**不是这六种 dtype 本身缺少内核**。其余六种未纳入本重载考察。NPU 侧该算子没有内核、走 CPU 兜底，故两行一致。

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、输出形状与 `self` 完全一致，扫描不改变形状；`dim` 指向的维度可任意长。


### 2.3 特殊限制说明

1、`self` 必须携带与 `dim` 匹配的具名维度。ATen 实现入口是 `dimname_to_position(self, dim)`，找不到该名字会直接抛错；换言之须先 `refine_names` 或经具名张量路径构造输入。

2、Named Tensor 在社区侧不成熟，inductor 当前不支持，具名张量进图即报：

```
aten::new_empty is not yet supported with named tensors. Please drop names via
`tensor = tensor.rename(None)`
```

### 2.4 使用方法

暂无可用测试用例。具名张量进图会被 inductor 直接拒绝（见 2.3 第 2 条），且 NPU 侧无该算子内核、走 CPU 兜底（见 2.3 第 4 条），两条路都拿不到可校验的 device kernel。
