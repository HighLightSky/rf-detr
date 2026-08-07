# 正式对比（注意 common.py 里 EPOCHS=1 是流水线测试值，正式用 --epochs 覆盖）
python src/scripts/test_sga/run.py --variant baseline
python src/scripts/test_sga/run.py --variant spm_only   # 已改残差融合
python src/scripts/test_sga/run.py --variant fixed_sga_lb
python src/scripts/test_sga/run.py --variant fixed_sga_res
python src/scripts/test_sga/run.py --variant attn_bias
