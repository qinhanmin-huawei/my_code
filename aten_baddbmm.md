# aten.baddbmm

## 1. OP 概述
### 1.1 算子原型
```
torch.ops.aten.baddbmm(
    self: Tensor, 
    batch1: Tensor, 
    batch2: Tensor, 
    beta: Scalar = 1, 
    alpha: Scalar = 1
) -> Tensor
```
### 1.2 算子功能
`aten.baddbmm` 执行批量矩阵乘加偏置操作：对批量维上每个批次执行矩阵乘法，并按比例加和到 `self` 上。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `self` | `Tensor` | 三维批量偏置张量，形状为 `(B, M, N)` |
| `batch1` | `Tensor` | 三维批量左乘矩阵，形状为 `(B, M, K)` |
| `batch2` | `Tensor` | 三维批量右乘矩阵，形状为 `(B, K, N)` |
| `beta` | `Scalar` | `self` 的缩放系数，默认为 1 |
| `alpha` | `Scalar` | 矩阵乘积结果的缩放系数，默认为 1 |
| 返回值 | `Tensor` | 批量矩阵乘加偏置的结果，形状为 `(B, M, N)` |

### 2.2 支持规格

#### 2.2.1 DataType 支持

| |uint8 | int8 | uint16 | int16 | uint32| int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| --- | --- | --- |  ---  |  --- |  --- | --- |  ---  |  --- | --- | --- | --- | --- |
| GPU | - | × | - | - | - | × | - | × | √ | √ | √ | × |
| NPU | - | × | - | - | - | × | - | × | √ | √ | √ | × |

#### 2.2.2 Shape 支持

1、输入为三维批量矩阵，`self` 形状 `(B, M, N)`、`batch1` 形状 `(B, M, K)`、`batch2` 形状 `(B, K, N)`

2、三个张量的批量维度 `B` 必须一致，`batch1.shape[2]` 必须等于 `batch2.shape[1]`

### 2.3 特殊限制说明

> 三个输入张量的批量维必须一致，且矩阵内维度必须匹配。`self` 的形状必须与矩阵乘积结果形状一致。

### 2.4 使用方法

```python
    dtypes = ['float16', 'bfloat16', 'float32']
    shapes = [(128, 1024, 4096)]

    @parametrize('shape', shapes)
    @parametrize('dtype', dtypes)
    def test_aten_baddbmm(self, shape, dtype):
        def op_calc(bias, batch1, batch2, beta, alpha):
            y = torch.ops.aten.baddbmm(bias, batch1, batch2, beta = beta, alpha = alpha)
            return y
        def op_calc_1(bias, batch1, batch2, beta, alpha):
            y = torch.ops.aten.baddbmm(bias, batch1, batch2, beta = beta, alpha = alpha)
            return y + y

        b = shape[0]
        m = shape[1]
        n = shape[2]
        k = shape[1]
        bias = self._generate_tensor((b, n, k), dtype)
        batch1 = self._generate_tensor((b, n, m), dtype)
        batch2 = self._generate_tensor((b, m, k), dtype)
        alpha = 1.5
        beta = 0.8
        self.execute_inductor_test(shape, dtype, op_calc, op_calc_1, bias, batch1, batch2, beta, alpha)
```
