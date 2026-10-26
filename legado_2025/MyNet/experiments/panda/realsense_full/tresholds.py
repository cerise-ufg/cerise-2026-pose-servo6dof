import pandas as pd
import matplotlib.pyplot as plt

# Ler o CSV
df = pd.read_csv("experiments/panda/realsense_full/results_formatado_complete.csv")

# ---------- Selecionar a linha com maior Val/ADD_60_mm_realsense ----------
linha_max_test = df.loc[df["Val/ADD_60_mm_realsense"].idxmax()]
colunas_test = [
    "Val/ADD_1_mm_realsense",
    "Val/ADD_5_mm_realsense",
    "Val/ADD_10_mm_realsense",
    "Val/ADD_20_mm_realsense",
    "Val/ADD_40_mm_realsense",
    "Val/ADD_60_mm_realsense",
    "Val/ADD_80_mm_realsense",
    "Val/ADD_100_mm_realsense"
]
add_values_test = linha_max_test[colunas_test].values

# ---------- Selecionar a linha com maior Val/ADD_60_mm_train ----------
linha_max_train = linha_max_test  # mesma linha para treino
colunas_train = [
    "Val/ADD_1_mm_train",
    "Val/ADD_5_mm_train",
    "Val/ADD_10_mm_train",
    "Val/ADD_20_mm_train",
    "Val/ADD_40_mm_train",
    "Val/ADD_60_mm_train",
    "Val/ADD_80_mm_train",
    "Val/ADD_100_mm_train"
]
add_values_train = linha_max_train[colunas_train].values

# Valor de 60mm para treino e teste em porcentagem
val_60_train_pct = linha_max_train["Val/ADD_60_mm_train"] * 100
val_60_test_pct = linha_max_test["Val/ADD_60_mm_realsense"] * 100

# ---------- GRÁFICO COMPARATIVO TREINO E TESTE ----------
thresholds = [1, 5, 10, 20, 40, 60, 80, 100]

plt.figure(figsize=(8,5))
plt.plot(thresholds, add_values_train, label=f"Treino (60mm={val_60_train_pct:.2f}%)", linewidth=2)
plt.plot(thresholds, add_values_test, label=f"Teste (60mm={val_60_test_pct:.2f}%)", linewidth=2, color="orange")
plt.xlabel("Threshold (mm)")
plt.ylabel("ADD")
plt.title("ADD x Threshold")
plt.grid(True)
plt.legend()
plt.show()

# ---------- TABELA DE RESULTADOS ----------
print("\nLinha com maior Val/ADD_60_mm_train:")
print(linha_max_train[colunas_train])

print("\nLinha com maior Val/ADD_60_mm_realsense:")
print(linha_max_test[colunas_test])
