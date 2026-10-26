# Abrir o arquivo original
with open('experiments/panda/realsense_full/results_formatado.csv', 'r', encoding='utf-8') as f:
    content = f.read()

# Substituir '.' por ',' e ',' por ';'
content = content.replace(',', '.').replace(';', ',')

# Salvar em um novo arquivo
with open('experiments/panda/realsense_full/results_formatado_complete.csv', 'w', encoding='utf-8') as f:
    f.write(content)

print("Arquivo processado e salvo como 'results_formatado_complete.csv'.")
