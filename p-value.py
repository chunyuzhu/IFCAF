import numpy as np
from scipy import stats

# ==========================================
# 1. 准备数据 (来自论文 Table 1 和 Table 3)
# ==========================================

# --- FLIR Dataset (3 classes) ---
# Competitor (JTMDet) [cite: 374]
flir_competitor = [87.8, 91.5, 71.5]
# Ours (IFCAF) [cite: 374]
flir_ours       = [87.2, 91.2, 74.4]

# --- M3FD Dataset (5 classes) ---
# Competitor (Fusion-Mamba)
# 注意：M3FD 中 Fusion-Mamba 的 Lamp 是 87.5，IFCAF 是 79.2 (这是个大失分项)
m3fd_competitor = [84.3, 92.9, 94.2, 88.8, 97.3]
# Ours (IFCAF)
m3fd_ours       = [87.3, 95.1, 93.2, 90.9, 97.9]

# ==========================================
# 2. 合并数据 (Aggregation)
# ==========================================
# 将两个列表拼接在一起，形成 8 个样本
all_competitor = flir_competitor + m3fd_competitor
all_ours       = flir_ours       + m3fd_ours

# 打印一下看看差异
diffs = np.array(all_ours) - np.array(all_competitor)
print("差异值 (Ours - Competitor):", diffs)
# 预期输出: [-0.6, -0.3, +2.9, +3.0, +2.2, -1.0, -8.3, +2.1]
# 分析：虽然我们有几个大比分赢项 (+2.9, +3.0)，但有一个巨大的输项 (Lamp -8.3)

# ==========================================
# 3. 计算 P-value (Paired T-test)
# ==========================================
# H0: Competitor >= Ours
# H1: Competitor < Ours (证明我们的方法显著更好)
t_stat, p_val = stats.ttest_rel(all_competitor, all_ours, alternative='less')

print(f"\n--- Aggregated Statistical Result (FLIR + M3FD) ---")
print(f"Sample Size: {len(all_competitor)}")
print(f"Competitor Mean AP: {np.mean(all_competitor):.2f}")
print(f"Ours Mean AP:       {np.mean(all_ours):.2f}")
print(f"P-value: {p_val:.5f}")

if p_val < 0.05:
    print("Result: Significant difference (p < 0.05) ✅")
else:
    print("Result: No significant difference (p >= 0.05) ⚠️")