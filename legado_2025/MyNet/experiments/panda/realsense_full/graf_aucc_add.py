import pandas as pd
import matplotlib.pyplot as plt

# Defina o limite máximo de épocas
MAX_EPOCH = 400

# Ler o CSV
df = pd.read_csv("experiments/panda/realsense_full/results_formatado_complete.csv")

# Filtrar até o máximo de épocas
if "epoch" in df.columns:
    df = df[df["epoch"] <= MAX_EPOCH]

# Colunas que queremos
colunas_desejadas = ["epoch", "Val/AUC_ADD_train", "Val/AUC_ADD_dr", 
                     "Val/AUC_ADD_photo", "Val/AUC_ADD_realsense"]

# Garantir que só pegamos as colunas que realmente existem no CSV
colunas_existentes = [col for col in colunas_desejadas if col in df.columns]

# ---------- GRÁFICO ----------
if "epoch" in colunas_existentes:
    x = df["epoch"]
    for col in colunas_existentes:
        if col != "epoch":
            plt.plot(x, df[col], label=col.replace("Val/deptherror_","").replace("Val/AUC_ADD_",""))
else:
    for col in colunas_existentes:
        plt.plot(df[col], label=col)

plt.xlabel("Epoch" if "epoch" in colunas_existentes else "Índice")
plt.ylabel("AUC/ADD")
plt.title("Treinamento- Dados Reais")
plt.legend()
plt.grid(True)
plt.show()

# ---------- TABELA ----------
resultados = {}

for col in colunas_existentes:
    if col == "epoch":
        continue
    
    minimo = df[col].max()
    
    # Valor na época MAX_EPOCH (se existir)
    if "epoch" in df.columns and MAX_EPOCH in df["epoch"].values:
        valor_epoca_max = df.loc[df["epoch"] == MAX_EPOCH, col].values[0]
    else:
        valor_epoca_max = None
    
    resultados[col] = {
        "Mínimo": minimo,
        f"Valor na época {MAX_EPOCH}": valor_epoca_max
    }

# Criar DataFrame com resultados
tabela_resultados = pd.DataFrame(resultados).T
print(tabela_resultados)
