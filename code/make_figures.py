import matplotlib.pyplot as plt
import numpy as np

# ── Figure 1: Accuracy Comparison (FedAvg vs FedProx vs FedProx+DP) ──
rounds_10 = list(range(1, 11))
fedavg_acc = [0.5583, 0.5583, 0.5583, 0.5583, 0.5583, 0.5583, 0.5667, 0.6250, 0.7500, 0.7250]
fedprox_acc = [0.4417, 0.4417, 0.4417, 0.4417, 0.5917, 0.7583, 0.8083, 0.8333, 0.8167, 0.8250]

rounds_20 = list(range(1, 21))
fedprox_dp_acc = [0.5583, 0.5583, 0.5583, 0.5583, 0.5583, 0.5583, 0.5583, 0.5583, 0.5583, 0.5583,
                  0.5583, 0.5667, 0.6500, 0.7167, 0.7417, 0.7500, 0.7583, 0.7750, 0.7833, 0.7833]

plt.figure(figsize=(10, 6))
plt.plot(rounds_10, fedavg_acc, marker='o', label='FedAvg (Baseline)', linewidth=2)
plt.plot(rounds_10, fedprox_acc, marker='s', label='FedProx (μ=0.1)', linewidth=2)
plt.plot(rounds_20, fedprox_dp_acc, marker='^', label='FedProx + Adaptive DP', linewidth=2)
plt.xlabel('Communication Round', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy vs Communication Rounds', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/fig1_accuracy_comparison.png', dpi=300)
plt.close()
print("Saved: fig1_accuracy_comparison.png")

# ── Figure 2: Privacy Budget (Epsilon) vs Rounds ──
epsilon_values = [5.7986, 6.0224, 6.2615, 6.5175, 6.7918, 7.0864, 7.4031, 7.7444, 8.1127, 8.5109,
                  8.9423, 9.4106, 9.9202, 10.4759, 11.0833, 11.7490, 12.4804, 13.2863, 14.1769, 15.1642]

plt.figure(figsize=(10, 6))
plt.plot(rounds_20, epsilon_values, marker='o', color='darkred', linewidth=2)
plt.axhline(y=10, color='green', linestyle='--', label='Target threshold (ε=10)')
plt.xlabel('Communication Round', fontsize=12)
plt.ylabel('Privacy Budget (ε)', fontsize=12)
plt.title('Privacy Budget Consumption Over Training Rounds', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/fig2_privacy_budget.png', dpi=300)
plt.close()
print("Saved: fig2_privacy_budget.png")

# ── Figure 3: Federated SHAP Feature Importance ──
features = ['exang', 'cp', 'sex', 'thalach', 'oldpeak', 'restecg', 'fbs', 'chol', 'age', 'trestbps']
shap_values = [0.2914, 0.0450, 0.0337, 0.0335, 0.0172, 0.0101, 0.0043, 0.0023, 0.0022, 0.0020]

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
plt.barh(features[::-1], shap_values[::-1], color=colors)
plt.xlabel('Mean |SHAP Value|', fontsize=12)
plt.title('Global Federated SHAP Feature Importance', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('../results/fig3_federated_shap.png', dpi=300)
plt.close()
print("Saved: fig3_federated_shap.png")

# ── Figure 4: Fairness Comparison ──
groups_gender = ['Cleveland\nFemale', 'Cleveland\nMale', 'Hungarian\nFemale', 'Hungarian\nMale']
acc_gender = [0.8144, 0.6796, 0.8642, 0.7934]

groups_age = ['Cleveland\nYoung', 'Cleveland\nMiddle', 'Cleveland\nOld',
               'Hungarian\nYoung', 'Hungarian\nMiddle', 'Hungarian\nOld']
acc_age = [0.8413, 0.6982, 0.6761, 0.8475, 0.8026, 0.8072]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].bar(groups_gender, acc_gender, color=['#FF9999', '#66B2FF', '#FF9999', '#66B2FF'])
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Fairness: Accuracy by Gender', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, 1)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axhline(y=0.05, color='gray', linestyle=':', alpha=0)

axes[1].bar(groups_age, acc_age, color=['#90EE90', '#FFD700', '#FF6961', '#90EE90', '#FFD700', '#FF6961'])
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Fairness: Accuracy by Age Group', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 1)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../results/fig4_fairness_comparison.png', dpi=300)
plt.close()
print("Saved: fig4_fairness_comparison.png")

print("\nAll figures saved to results/ folder!")