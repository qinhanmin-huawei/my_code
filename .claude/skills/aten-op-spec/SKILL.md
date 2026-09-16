---
name: aten-op-spec
description: 维护 AtenOp-Spec 仓库里的 aten 算子规格 markdown（一 overload 一文件）。用于按统一骨架新增/修改规格、回填或订正 GPU/NPU 的 DataType 支持表、把 1.2/2.3 的说明写简洁、从 atenop_test 的实测用例同步 2.4 代码块，并规避 CommonMark 段落塌陷导致的换行丢失。只改文档，不改测试代码。
---

# ATen 算子规格说明书

适用仓库：`/home/tangpx/Projects_work/aten_ops/AtenOp-Spec/`（362 个 `.md`，一个 overload 一个文件）。

结对的实测仓：`/home/tangpx/Projects_work/atenop_test/Aten_ops/`

- `test_aten_<op>.py` —— 2.4 代码块的来源
- `op_dtype_support.md` —— GPU/NPU dtype 结论的来源
- 那边的 skill `aten-op-test` 管测试代码，本 skill 管规格文档，两者互为镜像：改用例写法时，两边一起改。

## 硬约束

- **dtype 结论不能靠推断。** GPU/NPU 的格子只能来自 `op_dtype_support.md`，或用户在目标机上的实测输出。
  反例：按 CUDA kernel 的 `AT_DISPATCH_FLOATING_TYPES_AND2` 判定 erf/exp 对 int8/int32/int64/bool 是 `×`，
  但 `TensorIterator` 的 `promote_integer_inputs_to_float` 会先把整型提升到 float32 再分派，四种类型都正常出 float32。
  只看 kernel dispatch 会推错，以实测为准。
- **表格里不出现 `?`。** 不支持写 `-`；明确报错 / 未实现的写 `×`。也不要小写 `x` / `v`（仓库里已各有 18 / 1 处历史遗留，不要新增）。
- 用户在本批次的约定：**NPU 行照 GPU 行填**，纯标量重载也一样。老文件里两行不等的（如 `aten_abs.md` 的 bool）保持原样，不要去动。
- 只改被要求的文件与章节，不顺手统一其他文件的历史写法。

## 文档骨架

362 个文件中 361 个统一如下（`## 1.` 与 `### 1.1` 相邻不空行；其余各节之间是两个空行）：

```markdown
# aten.<op>.<overload>

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.<op>.<overload>(...) -> ...
```
### 1.2 算子功能
1、…（这个重载是什么）

2、…（补充：广播 / 输出形状 / 原地 / 常量折叠）


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self`/`input` | `Tensor` | … |
| 返回值 | `Tensor` | … |


### 2.2 支持规格

#### 2.2.1 DataType 支持

<表>

#### 2.2.2 Shape 支持

1、shape维度范围：[1-4]

2、…


### 2.3 特殊限制说明

> …


### 2.4 使用方法


```python
    …（4 空格缩进的方法体）
```
```

### 文件名

H1 限定名的每个 `.` 换成 `_`，再把连续的 3 个及以上下划线并成 2 个：

```
aten.__rshift__.Tensor_out        →  aten__rshift__Tensor_out.md
aten.bitwise_right_shift_.Tensor  →  aten_bitwise_right_shift__Tensor.md
```

仓库里有 11 个历史遗留不符合（`aten_mul_tensor.md`、`aten_remainder_int.md`、`aten_relu_.default.md`、`aten_frexp_tensor.md` …）。新文件按规则来，不要动老文件的名字。

## 各节写法

### 1.2 算子功能

两条，**短**。第一条一句话说清这个重载是什么（`>>` / `>>=` 的哪个重载、原地还是分配新张量、纯标量还是张量）；第二条只补一条信息（广播对齐、`out` 覆盖写、inductor 常量折叠）。

删掉「二进制位」「的**算术右移**」「专门针对…的情况」「该版本不会分配新的张量」这类冗词。参考成品：

```
1、`aten.__irshift__.Scalar` 是 `>>=` 运算符在 `other` 为标量时的原地版本：对 `self` 中每个整数按位算术右移 `other` 位（有符号整型高位补符号位），结果直接写回 `self`。
2、不分配新张量，返回修改后的 `self` 本身。
```

### 2.1 参数说明

一格一句话：`源张量，须为整型`、`位移量，须小于 self 位宽`、`新张量，形状与输入一致`。
不要写成「参与计算的源张量，须为整型（integral）类型」。

### 2.2.1 DataType 支持

三族表，按重载形态选，表头逐字照抄（含首格留空）：

| 族 | 表头 | 用在哪 |
| --- | --- | --- |
| 张量族 12 列 | `\|        \|uint8 \| int8 \| uint16 \| int16 \| uint32\| int32 \| uint64 \|  int64 \| fp16  \| fp32   \| bf16   \| bool   \|` | 张量重载 |
| 张量+复数 14 列 | 上一行再接 `complex64 \| complex128` | 支持复数的张量重载 |
| 纯标量 / complex 9 列 | `\|        \| int8 \| int32 \| int64 \| fp16 \| fp32 \| bf16 \| bool \| complex64 \| complex128 \|` | `float` / `int` / `Scalar` / `complex` 重载 |

表体两行，第一格写 `GPU` / `NPU`：

```markdown
| GPU | - | √ | - | … |
| NPU | - | √ | - | … |
```

- 9 列族的 `√` 指的是**承载结果的载体 tensor dtype**（2.4 里 `dtypes` 列表的那几种），不是标量参数自己的 dtype。表下补一句：
  `> 纯标量重载：参数固定为 int，返回 int，与 tensor dtype/shape 无关。表中 dtype 为承载结果的载体 tensor 类型。`
- 早期做法是让 dtype 表按 `a` / `b` / `返回值` 分行。纯标量 / complex 文件按用户要求已改成 GPU / NPU 两行，同时 2.1 参数说明保留原来的 `a` / `b` / 返回值 表。

### 2.2.2 Shape 支持

- 张量重载：`1、shape维度范围：[1-4]`；第二条写广播 / 输出形状。
- 纯标量重载：`不适用（无张量输入）。`（部分老文件写 `不适用，输入与返回值均为标量。`）

### 2.3 特殊限制说明

- 无限制就一行 `> 无`（116 个文件如此）。
- 多条用 `> 1、…` / `> 2、…`，**中间插一行只写 `>` 的空引用行**，否则渲染时连成一段。
- 通常一到两行讲两件事：数值合法域（位移量 < 位宽、除数非零、幂的底数），以及 dtype 边界（float 输入报 `xxx not implemented`）。成品：

```
> 位移量须小于 `self` 元素位宽（int8<8、int32<32、int64<64），否则为未定义行为（UB）。
> 仅支持整数 dtype：float 输入在 eager 报 `rshift_cpu not implemented`。
```

### 2.4 使用方法

- 代码块首行 4 空格缩进，只有方法体（353 个块全是这个形态，没有 `import` / `class` 行）。
- 开头两行字面量列表：`dtypes = [...]`（321 个文件）、`shapes = [...]`（259 个文件），shape 统一用 `[(128,), (128, 128), (128, 1024, 4096), (4, 8, 16, 32)]`。只有 15 个老文件用 `TestUtils._*`，不要推广。
- 两个图 `op_calc` / `op_calc_1`，收尾 `self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, ...)`。
- **代码主体从实测仓 copy**，外壳（字面量列表、方法名 = `test_` + 文件名去扩展名）按本仓约定，不逐字照搬 `TestUtils._*` 和仓库里的方法名。
- 结果变量命名有两派，跟着该文件 / 该批次既有写法，不要在同一文件里混：
  - `.out` 主流派：`out = torch.empty_like(...)` → `out=out` → `return out` / `return out + out`（49 个 `out=out`、33 个 `return out`）。
  - `y` 派（实测仓 skill 的要求）：`y = ...`、关键字参数写 `out=y`、`return y` / `return y + y`。
- 纯标量重载的载体，本仓主流写法（43 个文件）：

```python
    def op_calc(first_element, scalar_a):
        y = torch.ops.aten.<op>.<overload>(scalar_a, …)
        result = first_element + y
        return result

    def op_calc_1(first_element, scalar_a):
        y = torch.ops.aten.<op>.<overload>(scalar_a, …)
        result = first_element + y + y
        return result
```

  实测仓 skill 要求写成 `first_element * 2 + int(y)` 以避开 inductor 的 `remove_noop_ops` 删掉 `x + 0`；**规格这边按 `first_element + y` 写**，两边不一致是已知的，不要私自对调。

## CommonMark 排版陷阱

**连续的非空行会被并成同一段，换行塌成空格。** 全仓有 23 处纯文本、4 处引用块踩了这个坑。

- 编号条目 `1、` / `2、` 之间必须有空行（239 处已有空行）。
- 引用块内要分段，插一行只写 `>` 的行；只写两行 `>` 会渲染成一段。

改完用 `sed -n '/### 2.3/,/### 2.4/p' <file>` 之类的片段核对渲染效果，别只看整篇源码。

## 工作流程

1. 用户给一批文件时，**先用 `ls` / `git status` 确认清单**，别凭印象补文件或漏文件。
2. dtype / 数值结论先查 `op_dtype_support.md`；查不到就向用户要目标机实测结果，不要填推测值。
3. 逐文件改，改完把各文件的关键片段贴出来给用户看（用户常直接看图/表，不看整篇 diff）。
4. **不要自动 `git add` / `commit`。** 用户明确说「帮我 git add 一下」才 stage，且要排除他点名的文件（例：只 stage 那 16 个标量/complex 文件、不加 `rshift` 那一批）。提交前先问要不要开分支——仓库当前在 `main` 上。
