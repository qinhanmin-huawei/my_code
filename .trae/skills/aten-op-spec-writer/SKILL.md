---
name: aten-op-spec-writer
description: Write AtenOp operator specs into aten_op/spec_qhm from UT files in aten_op/pass_ut. Use when asked to write a spec for an aten or prims operator, or to document an op from its UT. Do not use for writing UT code.
---

# AtenOp Spec Writer

根据 `aten_op/pass_ut` 下的 UT 文件，按项目既定格式编写算子 spec，输出到 `aten_op/spec_qhm`。

## 输入与输出

- 输入：`aten_op/pass_ut/test_*.py`（用户可能给提交号，也可能直接给文件）
- 模板：`aten_op/spec_qhm` 下已有的同类 spec（如 `prims_ge.md`、`prims_ge_default.md`）
- 输出目录：以用户当次指定为准，默认 `aten_op/spec_qhm`
- 每个测试方法对应一个 md 文件

## 命名规则

op 名中的 `.` 一律替换为 `_`：

| 算子 | 文件名 |
| --- | --- |
| `prims.ge` | `prims_ge.md` |
| `prims.ge.default` | `prims_ge_default.md` |
| `aten.__and__.Tensor` | `aten___and___Tensor.md` |

## 编写步骤

1. 读 UT 文件，列出其中每个测试方法及其调用的算子（注意 `.default` overload 也各自成篇）。
2. 读同目录下最相近的已有 spec 作为模板，沿用其结构、措辞与表格。
3. 逐个生成 md，写完后回读抽查，确认结构与代码块完整。

## 文件结构

````markdown
# <op 全名，含命名空间>

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.<ns>.<op>(Tensor self, Tensor other) -> Tensor
```
### 1.2 算子功能
1、<一句话定位该算子>

2、<广播 / 比较 / 输出 dtype 说明>

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 第一个输入张量 |
| `other` | `Tensor` | 第二个输入张量，与 `self` 可广播 |
| 返回值 | `Tensor` | bool 张量，形状为广播后的 shape |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | ... |
| NPU | ... |

> <表格读法说明>

#### 2.2.2 Shape 支持

1、<广播或 shape 约束>

2、shape 维度范围：[1-4]

### 2.3 特殊限制说明

> <限制，或写“无”>

### 2.4 使用方法

```python
    dtypes = ['int8', 'int32', 'int64', 'bool', 'float16', 'bfloat16', 'float32']
    shapes = [(4,), (4, 8), (4, 8, 16), (4, 8, 16, 32)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_xxx(self, shape, dtype):
        <UT 原文>
```
````

## 硬性规则

1. 「2.4 使用方法」里的 `def` 函数必须与 UT **逐字一致** —— 方法名、参数、内部 `op_calc` / `op_calc_plus`、`_generate_tensor`、`execute_inductor_test` 都不能改。
2. 装饰器按模板风格改写（不是照抄 UT 的写法）：
   - UT 原文是 `@parametrize('shape', TestUtils._pointwise_demo_shapes)` 这类类属性引用；
   - md 里改成 `dtypes = [...]`、`shapes = [...]` 两个局部字面量，再写 `@parametrize('shape', shapes)`、`@parametrize('dtype', dtypes)`。
   - 一句话：**装饰器用模板写法，函数体用 UT 原文**。
3. 字面量取值必须查 `aten_op/pass_ut/testutils.py` 里 `TestUtils` 的真实定义（如 `_test_dtypes`、`_test_dtypes_without_bool`、`_pointwise_demo_shapes`），不要臆造。例如 `_test_dtypes` 是 7 个 dtype，`_test_dtypes_without_bool` 去掉 `bool`。
4. 数学公式用行内 `$...$`，不要用 `\( ... \)` —— 预览器不识别后者。
5. 不编造 DataType / Shape 支持情况：优先参照同类算子，存疑处在回复里向用户点明，请其确认。
6. UT 里的 `import`、`class` 定义、`instantiate_parametrized_tests(...)`、`if __name__ == "__main__"` 都不要写进 md。
7. 一轮任务涉及多个算子时，保持各文件之间除算子名、原型、功能描述、代码调用外的内容完全一致。
