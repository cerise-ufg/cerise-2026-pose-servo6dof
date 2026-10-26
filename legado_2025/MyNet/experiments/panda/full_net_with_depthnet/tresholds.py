import pandas as pd
import matplotlib.pyplot as plt

# Ler o CSV
df = pd.read_csv("experiments/panda/realsense_full/results_formatado_complete.csv")

# ---------- GRÁFICO TREINO ----------
# Encontrar a linha com maior Val/ADD_60_mm_train
linha_max_train = df.loc[df["Val/ADD_60_mm_train"].idxmax()]

# Selecionar colunas de interesse para treino
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

thresholds = [1, 5, 10, 20, 40, 60, 80, 100]
add_values_train = linha_max_train[colunas_train].values

plt.figure(figsize=(8,5))
plt.plot(thresholds, add_values_train, marker="o", label="Treino")
plt.xlabel("Threshold (mm)")
plt.ylabel("ADD")
plt.title("ADD x Threshold - Treino")
plt.grid(True)
plt.legend()
plt.show()

# ---------- GRÁFICO TESTE ----------
# Encontrar a linha com maior Val/ADD_60_mm_realsense
linha_max_test = df.loc[df["Val/ADD_60_mm_realsense"].idxmax()]

# Selecionar colunas de interesse para teste
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

plt.figure(figsize=(8,5))
plt.plot(thresholds, add_values_test, marker="o", color="orange", label="Teste")
plt.xlabel("Threshold (mm)")
plt.ylabel("ADD")
plt.title("ADD x Threshold - Teste")
plt.grid(True)
plt.legend()
plt.show()

# ---------- TABELA DE RESULTADOS ----------
print("\nLinha com maior Val/ADD_60_mm_train:")
print(linha_max_train[colunas_train])

print("\nLinha com maior Val/ADD_60_mm_realsense:")
print(linha_max_test[colunas_test])
