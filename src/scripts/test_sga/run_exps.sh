# 正式对比（注意 common.py 里 EPOCHS=10；正式用 --epochs 覆盖可留默认）
python src/scripts/test_sga/run.py --variant baseline
python src/scripts/test_sga/run.py --variant spm_only   # 已改残差融合
python src/scripts/test_sga/run.py --variant fixed_sga_lb
python src/scripts/test_sga/run.py --variant fixed_sga_res
python src/scripts/test_sga/run.py --variant attn_bias

# ── 多尺度 P3/P4 验证（0807 批次：E0 三变体 + E1 语义调制，输出 output/0807test_sga/）──
python src/scripts/test_sga/run.py --variant baseline_p4
python src/scripts/test_sga/run.py --variant vit_p3p4
python src/scripts/test_sga/run.py --variant spm_p3p4
python src/scripts/test_sga/run.py --variant semantic_film_p3p4
