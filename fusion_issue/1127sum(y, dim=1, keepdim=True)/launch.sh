rm -rf /tmp/torchinductor_root/*
rm -rf ~/.triton/dump
rm -rf ~/.triton/cache
export TORCH_COMPILE_DEBUG=1
TORCH_COMPILE_DEBUG_DIR=/data/q00887602/aten_op_test
export ASCEND_VISIBLE_DEVICES=1

echo $TORCH_COMPILE_DEBUG_DIR
# TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1
# inductor_generalize=normal pytest -sv script/fusion/pairwise/test_fusion_pairwise_P24.py -k test_fusion_pairwise_P24_shape3_dtype_float32 2>&1 | tee res.log
inductor_generalize=normal pytest -sv script/fusion/pairwise/test_fusion_pairwise_P23.py -k  test_fusion_pairwise_P23_shape12_dtype_float16 2>&1 | tee res.log
