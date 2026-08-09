# 2) 训练（每次只跑一个，输出目录自动分离 output/0810-SHWX-SemHead-<suffix>）：
# uv run src/scripts/ret-sscl/stage2_train.py p1     # 锚点基线（先跑，确定硬约束参照）
uv run src/scripts/ret-sscl/stage2_train.py e1a    # 完整语义头
uv run src/scripts/ret-sscl/stage2_train.py e2b    # 仅αS（掩码贡献 = e1a−e2b）
uv run src/scripts/ret-sscl/stage2_train.py e1c    # 仅M（语义方向贡献 = e1c−p1）
uv run src/scripts/ret-sscl/stage2_train.py e3b    # α冻结
uv run src/scripts/ret-sscl/stage2_train.py e3c    # novel类α更大
uv run src/scripts/ret-sscl/stage2_train.py e4b    # SSCL ω=1
uv run src/scripts/ret-sscl/stage2_train.py e4c    # 无SSCL

# 3) 8 个 run 全跑完后，批量评估 + 对比表 + 硬约束 PASS/FAIL：
uv run src/scripts/ret-sscl/eval_ablation.py
# 输出：output/0810-SHWX-SemHead-compare/{comparison.md, comparison.csv}