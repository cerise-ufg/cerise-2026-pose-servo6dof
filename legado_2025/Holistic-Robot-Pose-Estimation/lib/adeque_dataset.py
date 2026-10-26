import os
import shutil
from pathlib import Path
import threading


def process_folder(source_path, dest_path):
    """
    Função que será executada por cada thread para copiar os arquivos.
    """
    if not source_path.is_dir():
        return

    print(f"\nProcessando pasta: {source_path.name}")

    # Cria a subpasta correspondente no diretório de destino
    dest_path.mkdir(exist_ok=True)

    # Busca e ordena todos os arquivos .json
    json_files = sorted(list(source_path.glob("*.json")))

    # Inicializa os contadores para o novo processo
    num_pairs_copied = 0

    # Copia os primeiros 2000 pares de arquivos (json e jpg correspondente)
    for json_file in json_files[:5000]:
        # Obtém o nome base do arquivo (ex: '0001' de '0001.json')
        file_stem = json_file.stem

        # Constrói o caminho para o arquivo .rgb.jpg correspondente
        jpg_file = source_path / f"{file_stem}.rgb.jpg"
        if num_pairs_copied >= 2000:
            break
        # Verifica se o arquivo .rgb.jpg existe antes de copiar
        if jpg_file.exists():
            # Copia o arquivo .json
            shutil.copy(json_file, dest_path / json_file.name)

            # Copia o arquivo .rgb.jpg
            shutil.copy(jpg_file, dest_path / jpg_file.name)

            num_pairs_copied += 1

    print(f"  - Copiados {num_pairs_copied} pares de arquivos (.json e .rgb.jpg) para {source_path.name}.")


def main():
    # Defina os diretórios de origem e destino
    source_dir = Path("./data/dream/complete/synthetic")
    dest_dir = Path("./data/dream/reduce/synthetic")

    # Certifique-se de que o diretório de destino exista
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Buscando subpastas em: {source_dir}")

    threads = []
    # Itera sobre todas as subpastas e cria uma thread para cada uma
    for sub_dir in sorted(source_dir.iterdir()):
        if sub_dir.is_dir():
            dest_sub_dir = dest_dir / sub_dir.name

            # Cria a thread, passando a função e os argumentos
            thread = threading.Thread(target=process_folder, args=(sub_dir, dest_sub_dir))
            threads.append(thread)
            thread.start()

    # Aguarda todas as threads terminarem
    for thread in threads:
        thread.join()

    print("\nProcesso concluído!")


if __name__ == "__main__":
    main()