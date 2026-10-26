import pandas as pd
import matplotlib.pyplot as plt

# Defina o limite máximo de épocas
MAX_EPOCH = 400

# Ler o CSV
df = pd.read_csv("experiments/panda/full_net_with_depthnet/results.csv")

# Filtrar até o máximo de épocas
if "epoch" in df.columns:
    df = df[df["epoch"] <= MAX_EPOCH]

# Colunas que queremos
colunas_desejadas = ["epoch", "Val/loss_train", "Val/loss_dr", 
                     "Val/loss_photo", "Val/loss_realsense"]

# Garantir que só pegamos as colunas que realmente existem no CSV
colunas_existentes = [col for col in colunas_desejadas if col in df.columns]

# ---------- GRÁFICO ----------
if "epoch" in colunas_existentes:
    x = df["epoch"]
    for col in colunas_existentes:
        if col != "epoch":
            plt.plot(x, df[col], label=col.replace("Val/deptherror_","").replace("Val/loss_",""))
else:
    for col in colunas_existentes:
        plt.plot(df[col], label=col)

plt.xlabel("Epoch" if "epoch" in colunas_existentes else "Índice")
plt.ylabel("Loss")
plt.title("Treinamento- Dados Sintéticos")
plt.legend()
plt.grid(True)
plt.show()

# ---------- TABELA ----------
resultados = {}

for col in colunas_existentes:
    if col == "epoch":
        continue
    
    minimo = df[col].min()
    
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
