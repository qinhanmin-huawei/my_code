============================= test session starts ==============================
platform linux -- Python 3.11.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/local/python3.11.13/bin/python3.11
cachedir: .pytest_cache
hypothesis profile 'default' -> database=DirectoryBasedExampleDatabase('/data/q00887602/aten_op_test/.hypothesis/examples')
rootdir: /data/q00887602/aten_op_test
plugins: hypothesis-6.70.1, anyio-4.10.0
collecting ... collected 16 items / 15 deselected / 1 selected

script/fusion/pairwise/test_fusion_pairwise_P24.py::TestFusionPairwiseP24::test_fusion_pairwise_P24_shape3_dtype_float32 2026-11-27 01:35:51.348[INFO][testutils:351]ratio_result=================================== CASE TestFusionPairwiseP24 START ===================================
2026-11-27 01:35:51.348[INFO][testutils:271]clear env done: /tmp/torchinductor_root
2026-11-27 01:35:51.848[WARNING][testutils:268]~/.triton/dump 路径不存在
2026-11-27 01:35:51.848[WARNING][testutils:268]~/.triton/cache 路径不存在
2026-11-27 01:35:51.848[WARNING][testutils:268]/tmp/torchinductor_root 路径不存在
2026-11-27 01:35:51.848[WARNING][testutils:268]torch_compile_debug 路径不存在
2026-11-27 01:35:51.849[INFO][testutils:364]=-=-=-=-=-=-=-=-=-=-=-=- TestCase_ID: test_fusion_pairwise_P24.TestFusionPairwiseP24.test_fusion_pairwise_P24_shape3_dtype_float32 -=-=-=-=-=-=-=-=-=-=-=-=
2026-11-27 01:35:52.265[WARNING][testutils:268]/tmp/torchinductor_root 路径不存在
2026-11-27 01:35:52.265[WARNING][testutils:268]~/.triton/dump 路径不存在
2026-11-27 01:35:52.265[WARNING][testutils:268]~/.triton/cache 路径不存在
2026-11-27 01:35:52.265[WARNING][testutils:268]/tmp/torchinductor_root 路径不存在
2026-11-27 01:35:52.265[WARNING][testutils:268]torch_compile_debug 路径不存在
P24 2x7 NaN/inf: shape=(255, 257), dtype=float32
2026-11-27 01:36:03.949[INFO][testutils:489]eager-func success
2026-11-27 01:36:04.544[INFO][testutils:498]inductor-func compile success
W1127 01:36:18.996000 291 site-packages/torch/_inductor/debug.py:545] [0/0] model__0_inference_0 debug trace: torch_compile_debug/torchinductor/model__0_inference_0.0
2026-11-27 01:36:19.246[INFO][testutils:505]inductor-func success
2026-11-27 01:36:19.247[INFO][testutils:810]output_code files: ['torch_compile_debug/torchinductor/model__0_inference_0.0/output_code.py']
[W1127 01:36:19.602278064 ToKernelNpu.cpp:40] Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())
在第 80 行发现@triton.jit
std_result:
(tensor([        nan,         inf,        -inf,  36052.5312, -26978.0469,
        -23415.4121,  63815.0156, -33030.9961, -44015.5938,   1083.7148,
         24721.5781, -22684.9219, -27776.5391,   9672.3340,  34445.8828,
         39249.4492,  -6530.9834, -15792.4912, -17106.9922, -10364.1709,
        -40846.8203,  41921.0156,  -1046.2236,  29938.9023, -13101.3613,
        -36648.3477,  -3383.4473,    801.2090,  -1903.1289,  87029.1250,
        -17172.5234,  36725.0703,   -692.4746, -12248.3213,  17832.1406,
         20862.4414,  17624.2031,   -300.9316,   6490.0713,  -6511.6367,
         -8999.9961,  37100.6641,   4772.6211, -14694.7637, -26029.0703,
         30721.2324,  -5694.0059,  44792.7422,  -2470.6289,  36581.3750,
        -22722.8789, -28416.7383,  24483.7715,   9736.6006, -15051.2227,
         14861.9609,  33206.5938, -31916.2656, -40963.7383,  22330.9824,
         18081.8008,  61910.2461,  13389.6719, -67482.9297, -28799.5137,
         12043.2480, -49475.9141,  27505.0254, -12940.7129,  25295.0879,
         35078.5469, -31179.3418, -63725.1953,  11080.3242,  60127.8906,
         49357.2891,  11785.9570, -13968.6133,   -917.2295,   5968.8066,
        -58328.7031,  -7866.9434, -37668.2500,  49633.8164, -20361.0703,
        -63202.6133, -22815.6289,  43696.8359, -23211.9238, -13213.8047,
        -48305.3594, -28747.7656,  -5234.6401,  55134.5039, -23141.2305,
         43914.5625,   2353.9353,  42106.5781,  13617.9727,  33954.2031,
        -21379.9883,  47160.8008,  77172.3125, -31928.7266,  66451.8672,
          6445.5312, -14139.4043, -18002.6328,  54541.0859, -36119.7070,
         26253.3164,  49228.8516,  15248.0840,  13646.4053,  23661.6445,
        -14158.6426,  36620.7031, -53138.0117,  49707.6719,   2368.8184,
        -46012.0938,   3648.9531,   3721.5186, -24622.5547,  -6212.6855,
        -31314.2734, -47535.8672,  62381.6484, -42612.5000, -56472.4531,
        -27510.5059,  19785.5098, -12744.8477,  -9793.0713,  56238.7500,
         35078.3945, -17385.8496, -50189.0742,  16503.2344,  56740.1797,
        -35273.3359,  48015.0703, -18108.2227,  59956.7500,  32210.2578,
         33947.8516, -19714.4629,  -3387.3945,  39320.7422,  38488.4102,
         30040.0605, -20415.1992,  15043.0889,    296.7224,   6760.5244,
         -1491.8438, -12105.5479,   4679.3193, -14850.4609,   6952.6445,
         13027.8477,   7359.5439,   -395.8809,   8283.0801, -29911.4336,
        -19691.5840,   1816.4785,  -3820.3320, -18539.8828,   1831.2910,
        -11670.0957, -16860.2168,  36111.5234, -36411.5469,  92569.6250,
         23426.7109, 107971.4219,  -7329.9023,  33769.1797, -39927.0547,
        -14689.7852,  34865.7578, -41506.5664,  13235.5635,   5385.6606,
         -2847.0801, -12501.1348,  11916.1016,   4172.4609, -17766.7168,
         -4343.5186,  15524.6270, -39107.1719, -30981.0820,  15550.2930,
        -35128.6602,  39584.7109, -48225.2656,    802.9785,  35043.4141,
         -5514.6875,   8904.1074, -12425.1846,  18116.1367,  17247.3105,
         23098.5234, -11815.6250,  20070.9336, -11276.5527,  73317.3047,
        -24011.2812,  -3908.0352,  -4457.6133, -29747.7695,  -5104.8281,
         41770.8242,  65787.8125, -13538.6592, -76534.2188,  10540.1055,
         33158.8438,   6562.6113,  17905.1875,   4431.4883,  48928.8750,
         20901.1211, -38172.2812,  50751.8828, -34773.1758, -23481.6836,
         34934.8516,   1046.2676,  -3848.3906, -29279.2500, -17717.4609,
         -3844.0674, -50489.3125,  -6154.1465,   7525.6133, -32344.3984,
        -29569.2520,  38887.6484, -33360.2070, -52161.0977,  -7591.3330,
           552.9438, -48194.6875, -41712.0664,  20277.7539,  61422.1602,
        -10910.6787,  10502.1660, -20752.2773,   3362.8359,   6801.6875],
       device='npu:0'), tensor([      nan,       inf, 5471.2437, 6286.7349, 5272.0728, 4829.2070,
        7047.9790, 6519.7769, 5145.6196, 4981.7686, 5203.1182, 7392.7354,
        6368.1274, 5270.3960, 5344.6143, 6071.1797, 5328.3589, 5797.2822,
        5349.9360, 6371.0312, 5631.7930, 6482.3940, 5513.6641, 5701.2085,
        4779.6328, 5690.9800, 5311.2944, 5228.3633, 6068.7520, 5608.7993,
        4623.8901, 5913.6313, 4245.6011, 5772.9985, 5075.0376, 8001.0938,
        5529.2378, 5654.5854, 4760.7549, 7034.3872, 6121.6846, 4558.2949,
        6717.4595, 7371.1543, 5437.1504, 7221.6958, 5701.5264, 5187.5386,
        5759.8047, 5394.9219, 6510.5884, 5391.4590, 5506.8560, 6362.1685,
        6654.8896, 5573.3022, 5033.4233, 5023.1230, 4834.4507, 5870.4146,
        5803.5615, 6035.9067, 5476.0933, 4723.9746, 4979.8682, 4659.2368,
        5906.0566, 4242.7012, 5731.3481, 4926.9644, 4757.3506, 5485.4775,
        5577.7114, 4427.6431, 5374.6025, 5365.1426, 5055.4458, 6119.7275,
        5321.1772, 4890.9761, 5458.5825, 5736.5679, 6215.8335, 5731.5010,
        5582.6255, 4809.9214, 5530.0210, 5914.8120, 5677.4741, 4429.0190,
        6830.4336, 4017.5347, 5669.1929, 6958.4258, 5421.7114, 5487.9272,
        5752.3853, 5962.2256, 5046.4434, 6685.3647, 6380.7656, 6241.5186,
        5153.9785, 4767.3936, 6286.5454, 5734.8384, 6009.4512, 5283.5796,
        4666.2837, 5188.6377, 6777.2529, 6199.0137, 5780.0103, 5809.2476,
        5758.4653, 5085.5410, 6184.1489, 4682.0898, 7313.3921, 4856.4199,
        6958.8130, 4791.2456, 6386.8438, 4476.6733, 6227.2197, 5749.3379,
        4796.2051, 6034.4219, 6080.1187, 6115.9829, 5016.6006, 5537.4204,
        4730.8296, 4385.9883, 4982.1748, 6519.9932, 5071.3398, 6512.4434,
        7179.9790, 5242.4546, 4649.2988, 7462.5835, 5191.2891, 6764.5093,
        5584.1006, 5761.5635, 5070.8169, 4539.5405, 4786.6230, 6148.3833,
        5812.7959, 5569.2964, 5821.8623, 4433.7593, 5075.6470, 5015.8438,
        5752.1294, 5215.0537, 6063.7368, 5049.0527, 5679.5229, 5890.4458,
        4719.6841, 5942.2720, 7073.4995, 5246.2900, 5109.5381, 5981.3643,
        4099.0859, 6068.6392, 5752.7793, 5740.0205, 5343.3896, 5164.7842,
        6200.8770, 5164.0601, 5982.6626, 4962.4526, 7834.0713, 5281.5620,
        5596.1211, 6864.7891, 6899.5894, 6443.0908, 5532.5957, 4130.3115,
        5518.8184, 5041.8564, 6034.1436, 8153.5103, 6605.0547, 5700.7344,
        4280.8755, 4370.5762, 5628.8872, 5320.7275, 5414.9702, 5300.7593,
        5381.9448, 4928.2793, 5528.1777, 5607.1807, 4995.8447, 4424.9146,
        6068.4756, 5981.6943, 6172.2876, 5292.2539, 4250.8042, 6091.7808,
        7859.4336, 5487.5977, 6434.4243, 4116.7026, 4196.9106, 5619.6572,
        5190.4834, 6103.2524, 6079.2842, 6582.3994, 4953.5073, 5445.2607,
        6910.0405, 4686.0850, 6355.6465, 5253.0664, 5641.9072, 6221.8638,
        5004.3389, 5974.4399, 5394.0596, 4900.4429, 5786.5068, 6441.0269,
        7119.0044, 5743.1733, 5518.2598, 5443.1738, 5960.9854, 6013.9136,
        4950.4517, 5529.1650, 5922.3799, 4407.0146, 5220.6011, 4736.1494,
        5398.3467, 5864.2715, 5942.9233, 6269.2764, 5508.9219, 5548.6270,
        4915.8540, 5118.2954, 6809.0122], device='npu:0'))
inductor_result:
(tensor([        nan,         inf,        -inf,  36052.5234, -26978.0449,
        -23415.4121,  63815.0156, -33031.0039, -44015.5938,   1083.7129,
         24721.5801, -22684.9219, -27776.5391,   9672.3359,  34445.8867,
         39249.4492,  -6530.9844, -15792.4922, -17106.9922, -10364.1699,
        -40846.8281,  41921.0117,  -1046.2236,  29938.9004, -13101.3652,
        -36648.3438,  -3383.4438,    801.2080,  -1903.1289,  87029.1094,
        -17172.5215,  36725.0742,   -692.4746, -12248.3203,  17832.1406,
         20862.4375,  17624.2012,   -300.9287,   6490.0674,  -6511.6406,
         -8999.9951,  37100.6602,   4772.6201, -14694.7646, -26029.0684,
         30721.2344,  -5694.0039,  44792.7422,  -2470.6309,  36581.3750,
        -22722.8750, -28416.7344,  24483.7773,   9736.6045, -15051.2266,
         14861.9609,  33206.5938, -31916.2676, -40963.7422,  22330.9805,
         18081.8008,  61910.2500,  13389.6719, -67482.9375, -28799.5137,
         12043.2480, -49475.9141,  27505.0254, -12940.7148,  25295.0879,
         35078.5469, -31179.3438, -63725.1914,  11080.3242,  60127.8906,
         49357.2930,  11785.9570, -13968.6113,   -917.2266,   5968.8105,
        -58328.6992,  -7866.9434, -37668.2539,  49633.8203, -20361.0723,
        -63202.6094, -22815.6328,  43696.8398, -23211.9219, -13213.8018,
        -48305.3594, -28747.7637,  -5234.6396,  55134.5000, -23141.2305,
         43914.5625,   2353.9329,  42106.5781,  13617.9727,  33954.1953,
        -21379.9902,  47160.8008,  77172.3203, -31928.7266,  66451.8672,
          6445.5332, -14139.4053, -18002.6367,  54541.0820, -36119.7031,
         26253.3203,  49228.8477,  15248.0840,  13646.4062,  23661.6445,
        -14158.6406,  36620.7031, -53138.0156,  49707.6680,   2368.8193,
        -46012.0938,   3648.9453,   3721.5225, -24622.5508,  -6212.6895,
        -31314.2773, -47535.8672,  62381.6562, -42612.5039, -56472.4531,
        -27510.5098,  19785.5117, -12744.8467,  -9793.0723,  56238.7500,
         35078.3984, -17385.8477, -50189.0742,  16503.2344,  56740.1797,
        -35273.3359,  48015.0703, -18108.2227,  59956.7461,  32210.2578,
         33947.8516, -19714.4629,  -3387.3965,  39320.7461,  38488.4102,
         30040.0586, -20415.1973,  15043.0889,    296.7219,   6760.5225,
         -1491.8457, -12105.5498,   4679.3193, -14850.4648,   6952.6426,
         13027.8467,   7359.5439,   -395.8750,   8283.0781, -29911.4297,
        -19691.5859,   1816.4795,  -3820.3320, -18539.8809,   1831.2910,
        -11670.0977, -16860.2129,  36111.5195, -36411.5391,  92569.6250,
         23426.7109, 107971.4297,  -7329.9023,  33769.1836, -39927.0469,
        -14689.7812,  34865.7656, -41506.5664,  13235.5625,   5385.6572,
         -2847.0820, -12501.1328,  11916.1055,   4172.4609, -17766.7129,
         -4343.5195,  15524.6240, -39107.1719, -30981.0840,  15550.2949,
        -35128.6641,  39584.7109, -48225.2617,    802.9785,  35043.4141,
         -5514.6836,   8904.1094, -12425.1855,  18116.1367,  17247.3105,
         23098.5234, -11815.6270,  20070.9336, -11276.5547,  73317.3125,
        -24011.2793,  -3908.0312,  -4457.6152, -29747.7656,  -5104.8242,
         41770.8281,  65787.8125, -13538.6602, -76534.2188,  10540.1084,
         33158.8438,   6562.6152,  17905.1875,   4431.4863,  48928.8750,
         20901.1250, -38172.2773,  50751.8828, -34773.1797, -23481.6875,
         34934.8516,   1046.2627,  -3848.3984, -29279.2520, -17717.4609,
         -3844.0693, -50489.3086,  -6154.1455,   7525.6094, -32344.4043,
        -29569.2559,  38887.6523, -33360.1992, -52161.0977,  -7591.3311,
           552.9424, -48194.6797, -41712.0703,  20277.7539,  61422.1641,
        -10910.6855,  10502.1650, -20752.2773,   3362.8330,   6801.6895],
       device='npu:0'), tensor([      nan,       inf, 5471.2437, 6286.7349, 5272.0728, 4829.2070,
        7047.9790, 6519.7769, 5145.6196, 4981.7686, 5203.1182, 7392.7354,
        6368.1274, 5270.3960, 5344.6143, 6071.1797, 5328.3589, 5797.2822,
        5349.9360, 6371.0312, 5631.7930, 6482.3940, 5513.6641, 5701.2085,
        4779.6328, 5690.9800, 5311.2944, 5228.3633, 6068.7520, 5608.7993,
        4623.8901, 5913.6313, 4245.6011, 5772.9985, 5075.0376, 8001.0938,
        5529.2378, 5654.5854, 4760.7549, 7034.3872, 6121.6846, 4558.2949,
        6717.4595, 7371.1543, 5437.1504, 7221.6958, 5701.5264, 5187.5386,
        5759.8047, 5394.9219, 6510.5884, 5391.4590, 5506.8560, 6362.1685,
        6654.8896, 5573.3022, 5033.4233, 5023.1230, 4834.4507, 5870.4146,
        5803.5615, 6035.9067, 5476.0933, 4723.9746, 4979.8682, 4659.2368,
        5906.0566, 4242.7012, 5731.3481, 4926.9644, 4757.3506, 5485.4775,
        5577.7114, 4427.6431, 5374.6025, 5365.1426, 5055.4458, 6119.7275,
        5321.1772, 4890.9761, 5458.5825, 5736.5679, 6215.8335, 5731.5010,
        5582.6255, 4809.9214, 5530.0210, 5914.8120, 5677.4741, 4429.0190,
        6830.4336, 4017.5347, 5669.1929, 6958.4258, 5421.7114, 5487.9272,
        5752.3853, 5962.2256, 5046.4434, 6685.3647, 6380.7656, 6241.5186,
        5153.9785, 4767.3936, 6286.5454, 5734.8384, 6009.4512, 5283.5796,
        4666.2837, 5188.6377, 6777.2529, 6199.0137, 5780.0103, 5809.2476,
        5758.4653, 5085.5410, 6184.1489, 4682.0898, 7313.3921, 4856.4199,
        6958.8130, 4791.2456, 6386.8438, 4476.6733, 6227.2197, 5749.3379,
        4796.2051, 6034.4219, 6080.1187, 6115.9829, 5016.6006, 5537.4204,
        4730.8296, 4385.9883, 4982.1748, 6519.9932, 5071.3398, 6512.4434,
        7179.9790, 5242.4546, 4649.2988, 7462.5835, 5191.2891, 6764.5093,
        5584.1006, 5761.5635, 5070.8169, 4539.5405, 4786.6230, 6148.3833,
        5812.7959, 5569.2964, 5821.8623, 4433.7593, 5075.6470, 5015.8438,
        5752.1294, 5215.0537, 6063.7368, 5049.0527, 5679.5229, 5890.4458,
        4719.6841, 5942.2720, 7073.4995, 5246.2900, 5109.5381, 5981.3643,
        4099.0859, 6068.6392, 5752.7793, 5740.0205, 5343.3896, 5164.7842,
        6200.8770, 5164.0601, 5982.6626, 4962.4526, 7834.0713, 5281.5620,
        5596.1211, 6864.7891, 6899.5894, 6443.0908, 5532.5957, 4130.3115,
        5518.8184, 5041.8564, 6034.1436, 8153.5103, 6605.0547, 5700.7344,
        4280.8755, 4370.5762, 5628.8872, 5320.7275, 5414.9702, 5300.7593,
        5381.9448, 4928.2793, 5528.1777, 5607.1807, 4995.8447, 4424.9146,
        6068.4756, 5981.6943, 6172.2876, 5292.2539, 4250.8042, 6091.7808,
        7859.4336, 5487.5977, 6434.4243, 4116.7026, 4196.9106, 5619.6572,
        5190.4834, 6103.2524, 6079.2842, 6582.3994, 4953.5073, 5445.2607,
        6910.0405, 4686.0850, 6355.6465, 5253.0664, 5641.9072, 6221.8638,
        5004.3389, 5974.4399, 5394.0596, 4900.4429, 5786.5068, 6441.0269,
        7119.0044, 5743.1733, 5518.2598, 5443.1738, 5960.9854, 6013.9136,
        4950.4517, 5529.1650, 5922.3799, 4407.0146, 5220.6011, 4736.1494,
        5398.3467, 5864.2715, 5942.9233, 6269.2764, 5508.9219, 5548.6270,
        4915.8540, 5118.2954, 6809.0122], device='npu:0'))
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
2026-11-27 01:36:19.363[INFO][testutils:373]param_shape: (255, 257), param_dtype: float32, error_message: Result validation failed, used_func: func
2026-11-27 01:36:19.365[INFO][testutils:418]Target directory '/data/q00887602/aten_op_test/results_20261127_013551/output_code' is ready (created if it did not exist).
2026-11-27 01:36:19.365[INFO][testutils:424]File copy successful!
2026-11-27 01:36:19.365[INFO][testutils:425]  Source file: torch_compile_debug/torchinductor/model__0_inference_0.0/output_code.py
2026-11-27 01:36:19.365[INFO][testutils:426]  Target file: /data/q00887602/aten_op_test/results_20261127_013551/output_code/test_fusion_pairwise_P24_shape3_dtype_float32_(255, 257).py
2026-11-27 01:36:19.367[INFO][testutils:271]clear env done: /tmp/torchinductor_root
2026-11-27 01:36:19.867[WARNING][testutils:268]~/.triton/dump 路径不存在
2026-11-27 01:36:19.867[WARNING][testutils:268]~/.triton/cache 路径不存在
2026-11-27 01:36:19.867[WARNING][testutils:268]/tmp/torchinductor_root 路径不存在
2026-11-27 01:36:19.868[INFO][testutils:271]clear env done: torch_compile_debug
2026-11-27 01:36:20.368[INFO][testutils:433]Test results saved to: /data/q00887602/aten_op_test/results_20261127_013551/atenop_result_table.csv
FAILED2026-11-27 01:36:20.508[INFO][testutils:360]=================================== CASE TestFusionPairwiseP24 END ===================================
2026-11-27 01:36:20.508[INFO][testutils:361]accuracy_result:0/1,ratio_result:0/1


=================================== FAILURES ===================================
_____ TestFusionPairwiseP24.test_fusion_pairwise_P24_shape3_dtype_float32 ______

eager = (tensor([        nan,         inf,        -inf,  36052.5312, -26978.0469,
        -23415.4121,  63815.0156, -33030.996...3467, 5864.2715, 5942.9233, 6269.2764, 5508.9219, 5548.6270,
        4915.8540, 5118.2954, 6809.0122], device='npu:0'))
inductor = (tensor([        nan,         inf,        -inf,  36052.5234, -26978.0449,
        -23415.4121,  63815.0156, -33031.003...3467, 5864.2715, 5942.9233, 6269.2764, 5508.9219, 5548.6270,
        4915.8540, 5118.2954, 6809.0122], device='npu:0'))
eager_fn = <function TestFusionPairwiseP24.test_fusion_pairwise_P24.<locals>.fusion at 0x7f2d088de5c0>
args = (tensor([[       nan,  1136.9104, -1020.4494,  ...,  1707.5640, -3520.7458,
          1068.5684],
        [       inf,...7],
        [ 4311.8774,  1070.3722,  1119.5093,  ..., -1614.5294,  3692.0027,
          1476.0256]], device='npu:0'),)
kwargs = {'compiled_fn': <function TestFusionPairwiseP24.test_fusion_pairwise_P24.<locals>.fusion at 0x7f2cb2380680>}
cpu = (tensor([        nan,         inf,        -inf,  36052.5273, -26978.0430,
        -23415.4160,  63815.0078, -33031.003...4,
        5398.3467, 5864.2715, 5942.9233, 6269.2764, 5508.9219, 5548.6270,
        4915.8540, 5118.2954, 6809.0122]))

    def validator(eager, inductor, eager_fn=None, args=None, **kwargs):
        try:
>           _assert_nan_inf_close(eager, inductor)

script/fusion/pairwise/test_fusion_pairwise_P24.py:57: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

eager = (tensor([        nan,         inf,        -inf,  36052.5312, -26978.0469,
        -23415.4121,  63815.0156, -33030.996...3467, 5864.2715, 5942.9233, 6269.2764, 5508.9219, 5548.6270,
        4915.8540, 5118.2954, 6809.0122], device='npu:0'))
inductor = (tensor([        nan,         inf,        -inf,  36052.5234, -26978.0449,
        -23415.4121,  63815.0156, -33031.003...3467, 5864.2715, 5942.9233, 6269.2764, 5508.9219, 5548.6270,
        4915.8540, 5118.2954, 6809.0122], device='npu:0'))

    def _assert_nan_inf_close(eager, inductor):
        """NaN/±inf 位置 bitwise 一致，有限位置 allclose。"""
        for e, i in zip(eager, inductor):
            assert torch.isnan(e).equal(torch.isnan(i)), "NaN 位置不一致"
            assert torch.isposinf(e).equal(torch.isposinf(i)), "+inf 位置不一致"
            assert torch.isneginf(e).equal(torch.isneginf(i)), "-inf 位置不一致"
            mask = torch.isfinite(e)
>           torch.testing.assert_close(e[mask], i[mask], rtol=1e-5, atol=1e-5)
E           AssertionError: Tensor-likes are not close!
E           
E           Mismatched elements: 1 / 252 (0.4%)
E           Greatest absolute difference: 0.005859375 at index (159,) (up to 1e-05 allowed)
E           Greatest relative difference: 1.4801073120906949e-05 at index (159,) (up to 1e-05 allowed)

script/fusion/pairwise/test_fusion_pairwise_P24.py:28: AssertionError

During handling of the above exception, another exception occurred:

eager = (tensor([        nan,         inf,        -inf,  36052.5312, -26978.0469,
        -23415.4121,  63815.0156, -33030.996...3467, 5864.2715, 5942.9233, 6269.2764, 5508.9219, 5548.6270,
        4915.8540, 5118.2954, 6809.0122], device='npu:0'))
inductor = (tensor([        nan,         inf,        -inf,  36052.5234, -26978.0449,
        -23415.4121,  63815.0156, -33031.003...3467, 5864.2715, 5942.9233, 6269.2764, 5508.9219, 5548.6270,
        4915.8540, 5118.2954, 6809.0122], device='npu:0'))
eager_fn = <function TestFusionPairwiseP24.test_fusion_pairwise_P24.<locals>.fusion at 0x7f2d088de5c0>
args = (tensor([[       nan,  1136.9104, -1020.4494,  ...,  1707.5640, -3520.7458,
          1068.5684],
        [       inf,...7],
        [ 4311.8774,  1070.3722,  1119.5093,  ..., -1614.5294,  3692.0027,
          1476.0256]], device='npu:0'),)
kwargs = {'compiled_fn': <function TestFusionPairwiseP24.test_fusion_pairwise_P24.<locals>.fusion at 0x7f2cb2380680>}
cpu = (tensor([        nan,         inf,        -inf,  36052.5273, -26978.0430,
        -23415.4160,  63815.0078, -33031.003...4,
        5398.3467, 5864.2715, 5942.9233, 6269.2764, 5508.9219, 5548.6270,
        4915.8540, 5118.2954, 6809.0122]))

    def validator(eager, inductor, eager_fn=None, args=None, **kwargs):
        try:
            _assert_nan_inf_close(eager, inductor)
            return
        except AssertionError as eager_err:
            # 三方基准：eager vs inductor 不一致时，再看 inductor 是否与 CPU 一致。
            # 一致说明是 NPU eager 离群，而非 inductor 的问题。
            if eager_fn is None or args is None:
                raise
            try:
                cpu = eager_fn(*self.move_tensors_to_cpu(copy.deepcopy(args)))
            except Exception as cpu_err:
                raise AssertionError(
                    f"eager vs inductor 不一致，且 CPU 参考执行失败: "
                    f"{eager_err}; cpu: {cpu_err}"
                ) from eager_err
            try:
>               _assert_nan_inf_close(cpu, self.move_tensors_to_cpu(inductor))

script/fusion/pairwise/test_fusion_pairwise_P24.py:72: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

eager = (tensor([        nan,         inf,        -inf,  36052.5273, -26978.0430,
        -23415.4160,  63815.0078, -33031.003...4,
        5398.3467, 5864.2715, 5942.9233, 6269.2764, 5508.9219, 5548.6270,
        4915.8540, 5118.2954, 6809.0122]))
inductor = [tensor([        nan,         inf,        -inf,  36052.5234, -26978.0449,
        -23415.4121,  63815.0156, -33031.003...4,
        5398.3467, 5864.2715, 5942.9233, 6269.2764, 5508.9219, 5548.6270,
        4915.8540, 5118.2954, 6809.0122])]

    def _assert_nan_inf_close(eager, inductor):
        """NaN/±inf 位置 bitwise 一致，有限位置 allclose。"""
        for e, i in zip(eager, inductor):
            assert torch.isnan(e).equal(torch.isnan(i)), "NaN 位置不一致"
            assert torch.isposinf(e).equal(torch.isposinf(i)), "+inf 位置不一致"
            assert torch.isneginf(e).equal(torch.isneginf(i)), "-inf 位置不一致"
            mask = torch.isfinite(e)
>           torch.testing.assert_close(e[mask], i[mask], rtol=1e-5, atol=1e-5)
E           AssertionError: Tensor-likes are not close!
E           
E           Mismatched elements: 2 / 252 (0.8%)
E           Greatest absolute difference: 0.005126953125 at index (150,) (up to 1e-05 allowed)
E           Greatest relative difference: 1.727864582790062e-05 at index (150,) (up to 1e-05 allowed)

script/fusion/pairwise/test_fusion_pairwise_P24.py:28: AssertionError

The above exception was the direct cause of the following exception:

eager = (tensor([        nan,         inf,        -inf,  36052.5312, -26978.0469,
        -23415.4121,  63815.0156, -33030.996...3467, 5864.2715, 5942.9233, 6269.2764, 5508.9219, 5548.6270,
        4915.8540, 5118.2954, 6809.0122], device='npu:0'))
inductor = (tensor([        nan,         inf,        -inf,  36052.5234, -26978.0449,
        -23415.4121,  63815.0156, -33031.003...3467, 5864.2715, 5942.9233, 6269.2764, 5508.9219, 5548.6270,
        4915.8540, 5118.2954, 6809.0122], device='npu:0'))
eager_fn = <function TestFusionPairwiseP24.test_fusion_pairwise_P24.<locals>.fusion at 0x7f2d088de5c0>
args = (tensor([[       nan,  1136.9104, -1020.4494,  ...,  1707.5640, -3520.7458,
          1068.5684],
        [       inf,...7],
        [ 4311.8774,  1070.3722,  1119.5093,  ..., -1614.5294,  3692.0027,
          1476.0256]], device='npu:0'),)
kwargs = {'compiled_fn': <function TestFusionPairwiseP24.test_fusion_pairwise_P24.<locals>.fusion at 0x7f2cb2380680>}
cpu = (tensor([        nan,         inf,        -inf,  36052.5273, -26978.0430,
        -23415.4160,  63815.0078, -33031.003...4,
        5398.3467, 5864.2715, 5942.9233, 6269.2764, 5508.9219, 5548.6270,
        4915.8540, 5118.2954, 6809.0122]))

    def validator(eager, inductor, eager_fn=None, args=None, **kwargs):
        try:
            _assert_nan_inf_close(eager, inductor)
            return
        except AssertionError as eager_err:
            # 三方基准：eager vs inductor 不一致时，再看 inductor 是否与 CPU 一致。
            # 一致说明是 NPU eager 离群，而非 inductor 的问题。
            if eager_fn is None or args is None:
                raise
            try:
                cpu = eager_fn(*self.move_tensors_to_cpu(copy.deepcopy(args)))
            except Exception as cpu_err:
                raise AssertionError(
                    f"eager vs inductor 不一致，且 CPU 参考执行失败: "
                    f"{eager_err}; cpu: {cpu_err}"
                ) from eager_err
            try:
                _assert_nan_inf_close(cpu, self.move_tensors_to_cpu(inductor))
            except AssertionError as cpu_err:
>               raise AssertionError(
                    f"eager vs inductor 不一致，inductor vs cpu 也不一致: {cpu_err}"
                ) from cpu_err
E               AssertionError: eager vs inductor 不一致，inductor vs cpu 也不一致: Tensor-likes are not close!
E               
E               Mismatched elements: 2 / 252 (0.8%)
E               Greatest absolute difference: 0.005126953125 at index (150,) (up to 1e-05 allowed)
E               Greatest relative difference: 1.727864582790062e-05 at index (150,) (up to 1e-05 allowed)

script/fusion/pairwise/test_fusion_pairwise_P24.py:74: AssertionError

The above exception was the direct cause of the following exception:

self = <test_fusion_pairwise_P24.TestFusionPairwiseP24 testMethod=test_fusion_pairwise_P24_shape3_dtype_float32>
shape = (255, 257), dtype = 'float32'

    @parametrize("shape", _shapes())
    @parametrize('dtype', ['float32'])
    def test_fusion_pairwise_P24(self, shape, dtype):
        print(f"P24 2x7 NaN/inf: shape={shape}, dtype={dtype}")
    
        def fusion(x):
            s = torch.ops.aten.sum.dim_IntList(x, [1], False)
            m = torch.ops.aten.max.dim(x, 1, False)[0]
            return (s, m)
    
        def fusion_1(x):
            s, m = fusion(x)
            return (s + s, m + m)
    
        def validator(eager, inductor, eager_fn=None, args=None, **kwargs):
            try:
                _assert_nan_inf_close(eager, inductor)
                return
            except AssertionError as eager_err:
                # 三方基准：eager vs inductor 不一致时，再看 inductor 是否与 CPU 一致。
                # 一致说明是 NPU eager 离群，而非 inductor 的问题。
                if eager_fn is None or args is None:
                    raise
                try:
                    cpu = eager_fn(*self.move_tensors_to_cpu(copy.deepcopy(args)))
                except Exception as cpu_err:
                    raise AssertionError(
                        f"eager vs inductor 不一致，且 CPU 参考执行失败: "
                        f"{eager_err}; cpu: {cpu_err}"
                    ) from eager_err
                try:
                    _assert_nan_inf_close(cpu, self.move_tensors_to_cpu(inductor))
                except AssertionError as cpu_err:
                    raise AssertionError(
                        f"eager vs inductor 不一致，inductor vs cpu 也不一致: {cpu_err}"
                    ) from cpu_err
                print(
                    "P24: eager vs inductor 不一致，但 inductor 与 cpu 一致，"
                    f"判定为 NPU eager 离群: {eager_err}"
                )
    
        x = self._generate_tensor(shape, dtype).clone()
        # 分三行注入，避免 inf 与 -inf 在同一归约里相消
        x[0, 0] = float('nan')
        x[1, 0] = float('inf')
        x[2, 0] = float('-inf')
>       return self.execute_inductor_test(shape, dtype, fusion, fusion_1, x,
                                          op_name='aten.sum.dim_IntList',
                                          result_validator=validator,
                                          kernel_expect=-1)

script/fusion/pairwise/test_fusion_pairwise_P24.py:87: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <test_fusion_pairwise_P24.TestFusionPairwiseP24 testMethod=test_fusion_pairwise_P24_shape3_dtype_float32>
shape = (255, 257), dtype = 'float32'
func = <function TestFusionPairwiseP24.test_fusion_pairwise_P24.<locals>.fusion at 0x7f2d088de5c0>
func_plus = <function TestFusionPairwiseP24.test_fusion_pairwise_P24.<locals>.fusion_1 at 0x7f2d705bbe20>
op_name = 'aten.sum.dim_IntList', test_mode = 'default', cpu_reference = True
benchmark = 'profiler'

    def execute_inductor_test(
        self, shape, dtype, func, func_plus, *args, op_name=None,
        test_mode='default', cpu_reference=True, benchmark='profiler', synchronize=None,
        result_validator=None, kernel_expect=-1
    ):
        if test_mode not in ('default', 'non_kernel'):
            raise ValueError(f"Unknown test_mode: {test_mode}")
        if not isinstance(cpu_reference, bool):
            raise ValueError("cpu_reference must be a bool")
        if benchmark not in ('profiler', 'end_to_end', None):
            raise ValueError(f"Unknown benchmark: {benchmark}")
        if benchmark == 'end_to_end' and not callable(synchronize):
            raise ValueError("end_to_end timing requires a synchronize callback")
        if result_validator is not None and not callable(result_validator):
            raise ValueError("result_validator must be callable or None")
        if not isinstance(kernel_expect, int):
            raise ValueError("kernel_expect must be a int")
        self._test_mode = test_mode
        self.status = 'failure'
        def get_res_dtype(std_result):
            # 精度校验
            try:
                res_dtype = str(std_result.dtype).replace('torch.', '')
            except AttributeError:
                try:
                    res_dtype = str(std_result[0].dtype).replace('torch.', '')
                except (AttributeError, IndexError, TypeError):
                    # 如果还报错，中断执行
                    raise RuntimeError("无法获取数据类型")
            return res_dtype
    
        self.__class__.total_count += 1
        self.op_name = op_name
        # 在eager、inductor模式下执行op方法，并进行相关校验
        if not self.param_dtype:
            self.param_dtype = dtype
        self.param_shape = str(shape)
        args1 = copy.deepcopy(args)
        args2 = copy.deepcopy(args)
        # ========================================用func测试 ====================================================
        self.compare_target = 'eager'
        self.used_func = "func"
        try:
            std_result = func(*args1)
        except Exception as e:
            self.error_message = "eager failure"
            # raise AssertionError(f"eager failure: {e}") from e
            print(f"eager failure: {e}")
            try:
                args_cpu = self.move_tensors_to_cpu(copy.deepcopy(args))
                self.compare_target = 'cpu'
                std_result = func(*args_cpu)
            except Exception as e:
                raise AssertionError(f"eager failure and cpu failure: {e}") from e
        LOGGER.info('eager-func success')
        if self.isMmType:
            compiled_op_calc = torch.compile(func, backend="inductor",fullgraph=True,options={
                                                "max_autotune": True,
                                                "max_autotune_gemm_backends": "TRITON,ATEN",
                                                "epilogue_fusion": True,
                                            })
        else:
            compiled_op_calc = torch.compile(func, backend="inductor")
        LOGGER.info('inductor-func compile success')
        try:
            inductor_result = compiled_op_calc(*args2)
        except Exception as e:
            self.error_message = "inductor failure"
            raise AssertionError(f"inductor failure: {e}") from e
        # inductor 功能校验
        LOGGER.info('inductor-func success')
    
        self.check_compile_kernel(kernel_expect)
        if test_mode == 'non_kernel':
            if self.error_message == "FALLBACK":
                self.error_message = None
                LOGGER.info("non_kernel: Triton kernel is not required; inspect graphs/output code for compilation coverage.")
            else:
                raise AssertionError('The expected result is fallback, but it does not occur.')
        # ========================================用func_plus测试 =================================================
        if test_mode == 'default' and self.error_message in ("FALLBACK", "[failure]no output_code generated", 'kernel count error'):
            LOGGER.info('func_plus-test start')
            self.compare_target = 'eager'
            args1 = copy.deepcopy(args)
            args2 = copy.deepcopy(args)
            # 删除func生成的output_code等
            if "TORCH_COMPILE_DEBUG_DIR" in os.environ:
                target_path = os.getenv("TORCH_COMPILE_DEBUG_DIR", '')
            else:
                target_path = "torch_compile_debug"
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            # 在eager、inductor模式下执行op方法，并进行相关校验
            self.used_func = "func_plus"
            try:
                std_result = func_plus(*args1)
                # pass
            except Exception as e:
                self.error_message = "eager failure"
                # raise AssertionError(f"eager failure: {e}") from e
                print(f"eager failure: {e}")
                try:
                    args_cpu = self.move_tensors_to_cpu(copy.deepcopy(args))
                    self.compare_target = 'cpu'
                    std_result = func_plus(*args_cpu)
                except Exception as e:
                    raise AssertionError(f"eager failure and cpu failure: {e}") from e
            LOGGER.info('eager-func_plus success')
            if self.isMmType:
                compiled_op_calc = torch.compile(func_plus, backend="inductor",fullgraph=True,options={
                                                    "max_autotune": True,
                                                    "max_autotune_gemm_backends": "TRITON,ATEN",
                                                    "epilogue_fusion": True,
                                                })
            else:
                compiled_op_calc = torch.compile(func_plus, backend="inductor")
            LOGGER.info('inductor-func_plus compile success')
            try:
                inductor_result = compiled_op_calc(*args2)
            except Exception as e:
                self.error_message = "inductor failure"
                raise AssertionError(f"inductor failure: {e}") from e
            LOGGER.info('inductor-func_plus success')
            # inductor 功能校验
            self.check_compile_kernel(kernel_expect)
            if self.error_message == "FALLBACK" or self.error_message == "[failure]no output_code generated":
                if self.isMmType == True:
                    LOGGER.warning('The mm class atenop may fallback due to performance')
                else:
                    raise AssertionError(f"CAN NOT lowering!!!!!!!!!!!")
            elif self.error_message == 'kernel count error':
                raise AssertionError('The number of kernels does not meet expectations!!!')
            # 精度校验
        print(f"std_result:\n{std_result}")
        print(f"inductor_result:\n{inductor_result}")
        res_dtype = get_res_dtype(std_result)
        # self.accuracy_validate(std_result, inductor_result, res_dtype)
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        if self.used_func == "func":
            last_func = func
        elif self.used_func == "func_plus":
            last_func = func_plus
        else:
            raise AssertionError(f"used_func : {self.used_func} , is invalid")
    
        if self.compare_target == 'cpu':
            try:
                if result_validator is not None:
                    try:
                        result_validator(std_result, self.move_tensors_to_cpu(inductor_result), eager_fn=last_func, compiled_fn=compiled_op_calc, args=copy.deepcopy(args))
                    except:
                        self.accuracy_validate(std_result, self.move_tensors_to_cpu(inductor_result), res_dtype)
            except AssertionError as e:
                print(f"inductor vs cpu fail")
                self.error_message = "Precision verification failed"
                raise AssertionError(f"Inductor-CPU Precision verification failed: {e}") from e
            self.status = 'success'
            self.__class__.success_count += 1
            self.ratio_result = 'PASS'
            self.__class__.profiling_success_count += 1
            return
    
        if result_validator is not None:
            try:
                result_validator(std_result, inductor_result, eager_fn=last_func, compiled_fn=compiled_op_calc, args=copy.deepcopy(args))
            except Exception as e:
                self.error_message = "Result validation failed"
>               raise AssertionError(f"Result validation failed: {e}") from e
E               AssertionError: Result validation failed: eager vs inductor 不一致，inductor vs cpu 也不一致: Tensor-likes are not close!
E               
E               Mismatched elements: 2 / 252 (0.8%)
E               Greatest absolute difference: 0.005126953125 at index (150,) (up to 1e-05 allowed)
E               Greatest relative difference: 1.727864582790062e-05 at index (150,) (up to 1e-05 allowed)
E               
E               To execute this test, run the following from the base repo dir:
E                   python test_fusion_pairwise_P24.py TestFusionPairwiseP24.test_fusion_pairwise_P24_shape3_dtype_float32
E               
E               This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0

testutils.py:602: AssertionError
=============================== warnings summary ===============================
../../../usr/local/python3.11.13/lib/python3.11/site-packages/triton/language/extra/cann/extension/math_ops.py:13
  /usr/local/python3.11.13/lib/python3.11/site-packages/triton/language/extra/cann/extension/math_ops.py:13: FutureWarning: triton.language.extra.cann.extension.atan2 is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.atan2 instead.
    @_deprecated(replacement="triton.language.extra.cann.libdevice.atan2")

../../../usr/local/python3.11.13/lib/python3.11/site-packages/triton/language/extra/cann/extension/math_ops.py:35
  /usr/local/python3.11.13/lib/python3.11/site-packages/triton/language/extra/cann/extension/math_ops.py:35: FutureWarning: triton.language.extra.cann.extension.isfinited is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.isfinited instead.
    @_deprecated(replacement="triton.language.extra.cann.libdevice.isfinited")

../../../usr/local/python3.11.13/lib/python3.11/site-packages/triton/language/extra/cann/extension/math_ops.py:49
  /usr/local/python3.11.13/lib/python3.11/site-packages/triton/language/extra/cann/extension/math_ops.py:49: FutureWarning: triton.language.extra.cann.extension.finitef is deprecated and will be removed in the next release; use triton.language.extra.cann.libdevice.finitef instead.
    @_deprecated(replacement="triton.language.extra.cann.libdevice.finitef")

script/fusion/pairwise/test_fusion_pairwise_P24.py::TestFusionPairwiseP24::test_fusion_pairwise_P24_shape3_dtype_float32
  /usr/local/python3.11.13/lib/python3.11/site-packages/torch_npu/utils/storage.py:61: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
    if self.device.type == "npu":

script/fusion/pairwise/test_fusion_pairwise_P24.py: 14 warnings
  /usr/local/python3.11.13/lib/python3.11/site-packages/torch/jit/_script.py:365: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

script/fusion/pairwise/test_fusion_pairwise_P24.py::TestFusionPairwiseP24::test_fusion_pairwise_P24_shape3_dtype_float32
  /usr/local/python3.11.13/lib/python3.11/site-packages/triton/backends/ascend/compiler.py:1666: FutureWarning: Ascend environment variable 'TRITON_DISABLE_FFTS' is deprecated and will be removed in a future release; it is ignored; FFTS policy is derived from the explicit target architecture.
    _warn_deprecated_ascend_env_vars()

script/fusion/pairwise/test_fusion_pairwise_P24.py::TestFusionPairwiseP24::test_fusion_pairwise_P24_shape3_dtype_float32
  /usr/local/python3.11.13/lib/python3.11/site-packages/triton/backends/ascend/compiler.py:1679: FutureWarning: Ascend compile option 'enable_vf_fusion' is deprecated and will be removed in a future release; it is ignored; this switch is no longer forwarded to the NPU compiler.
    normalized_opts = opts if internal_options else _remove_deprecated_npu_options(opts)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED script/fusion/pairwise/test_fusion_pairwise_P24.py::TestFusionPairwiseP24::test_fusion_pairwise_P24_shape3_dtype_float32
================ 1 failed, 15 deselected, 20 warnings in 32.29s ================
