rm -rf /tmp/torchinductor_root/*
rm -rf ~/.triton/dump
rm -rf ~/.triton/cache
export TORCH_COMPILE_DEBUG=1
TORCH_COMPILE_DEBUG_DIR=/data/q00887602/aten_op_test
export ASCEND_VISIBLE_DEVICES=1

echo $TORCH_COMPILE_DEBUG_DIR

inductor_generalize=normal pytest -sv script/fusion/base/new_base/test_fusion_base_11_cat_clamp_mm.py::TestFusionBase11CatClampMm::test_fusion_base_11_cat_clamp_mm_shape1_dtype_bfloat16 2>&1 | tee res.log
