import pandas as pd
import matplotlib.pyplot as plt

max_epoch = 500

# Ler o CSV
df = pd.read_csv("experiments/panda/depth_net/results.csv")

# Colunas desejadas
colunas_desejadas = ["epoch", "Val/loss_train", "Val/deptherror_photo", 
                     "Val/deptherror_dr", "Val/deptherror_realsense"]
colunas_existentes = [col for col in colunas_desejadas if col in df.columns]

# Preparar eixo X
if "epoch" in colunas_existentes:
    x = df["epoch"][:max_epoch]
else:
    x = range(min(len(df), max_epoch))

# Plotar as métricas
plt.figure(figsize=(10,6))
for col in colunas_existentes:
    if col != "epoch":
        y = df[col][:max_epoch]
        nome_legenda = col.replace("Val/deptherror_", "Erro ").replace("Val/loss_", "Erro ")
        plt.plot(x, y, label=nome_legenda)

plt.xlabel("Epoch" if "epoch" in colunas_existentes else "Índice")
plt.ylabel("Valor")
plt.title("DepthNet - Métricas de Validação")
plt.legend()
plt.grid(True)
plt.show()

# ---------- TABELA DE RESULTADOS ----------
resultados = {}
for col in colunas_existentes:
    if col == "epoch":
        continue
    minimo = df[col].min()
    
    if "epoch" in df.columns and 700 in df["epoch"].values:
        valor_epoca_700 = df.loc[df["epoch"] == 700, col].values[0]
    else:
        valor_epoca_700 = df[col].iloc[-1]  # último valor disponível caso a época 700 não exista
    
    resultados[col] = {
        "Mínimo": minimo,
        "Valor na época 700": valor_epoca_700
    }

tabela_resultados = pd.DataFrame(resultados).T
print(tabela_resultados)
