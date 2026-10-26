import os
import shutil
from pathlib import Path
from typing import List


# --- Método para limpar arquivos JSON sem par ---
def limpar_json_sem_par(caminho_do_diretorio: Path):
    """
    Verifica se cada arquivo .json tem um .rgb.jpg correspondente
    no mesmo diretório. Se não tiver, o arquivo .json é apagado.

    Args:
        caminho_do_diretorio (Path): O caminho para o diretório a ser verificado e limpo.
    """
    if not caminho_do_diretorio.is_dir():
        print(f"Erro: O caminho '{caminho_do_diretorio}' não é um diretório válido.")
        return

    print(f"Iniciando a limpeza de arquivos JSON sem par em: {caminho_do_diretorio}")
    arquivos_apagados = 0

    for json_file in caminho_do_diretorio.glob("*.json"):
        file_stem = json_file.stem
        jpg_file = caminho_do_diretorio / f"{file_stem}.rgb.jpg"

        if not jpg_file.exists():
            print(f"AVISO: O arquivo {json_file.name} não tem um .rgb.jpg correspondente. Apagando...")
            json_file.unlink()  # O método unlink() é usado para apagar o arquivo
            arquivos_apagados += 1

    print(f"Limpeza concluída. Total de arquivos JSON apagados: {arquivos_apagados}")


# --- Método para verificar arquivos JPG sem par ---
def verificar_jpg_sem_par(caminho_do_diretorio: Path) -> List[Path]:
    """
    Verifica se cada arquivo .rgb.jpg tem um arquivo .json correspondente.

    Args:
        caminho_do_diretorio (Path): O caminho para o diretório a ser verificado.

    Returns:
        List[Path]: Uma lista com os caminhos completos dos arquivos .rgb.jpg que
                    não têm um .json correspondente.
    """
    if not caminho_do_diretorio.is_dir():
        print(f"Erro: O caminho '{caminho_do_diretorio}' não é um diretório válido.")
        return []

    print(f"\nVerificando arquivos JPG sem par em: {caminho_do_diretorio}")
    arquivos_faltando = []

    for jpg_file in caminho_do_diretorio.glob("*.rgb.jpg"):
        file_stem = jpg_file.stem
        json_file = caminho_do_diretorio / f"{file_stem}.json"

        if not json_file.exists():
            print(f"AVISO: O arquivo {jpg_file.name} não tem um .json correspondente.")
            arquivos_faltando.append(jpg_file)


    if not arquivos_faltando:
        print("\nVerificação de JPGs concluída: Todos os arquivos .rgb.jpg têm um .json correspondente.")

    return arquivos_faltando


# --- Exemplo de uso do script: ---
if __name__ == "__main__":
    caminho_da_pasta = Path("./data/dream/real/panda-3cam_realsense")

    # Passo 1: Limpar os JSONs sem par
    limpar_json_sem_par(caminho_da_pasta)

    print("-" * 50)  # Linha divisória para separar as duas etapas

    # Passo 2: Verificar os JPGs que ficaram sem par
    arquivos_jpg_incompletos = verificar_jpg_sem_par(caminho_da_pasta)

    if arquivos_jpg_incompletos:
        print(
            f"\nResumo da verificação de JPGs: Encontrados {len(arquivos_jpg_incompletos)} arquivos .rgb.jpg sem par.")
        for arquivo in arquivos_jpg_incompletos:
            print(f"- {arquivo}")