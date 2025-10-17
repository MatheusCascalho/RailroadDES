import re
import ast

def extrair_listas_best_solution(caminho_arquivo):
    padrao = re.compile(r"Current Best Solution:\s*(\[.*\])")
    listas_encontradas = []

    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        for linha in f:
            correspondencia = padrao.search(linha)
            if correspondencia:
                try:
                    # Converte a string da lista para uma lista Python real
                    lista = ast.literal_eval(correspondencia.group(1))
                    listas_encontradas.append(lista)
                except Exception as e:
                    print(f"Erro ao converter linha: {linha.strip()} -> {e}")

    return listas_encontradas


# Exemplo de uso:
if __name__ == "__main__":
    caminho = "logs/vns/vns_FINAL.log"  # substitua pelo caminho do seu arquivo
    resultados = extrair_listas_best_solution(caminho)
    print(f"{len(resultados)} listas encontradas:")
    for i, lst in enumerate(resultados, 1):
        print(f"{i}: {lst}")
