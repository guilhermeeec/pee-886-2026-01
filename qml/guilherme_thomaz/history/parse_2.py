import re

from matplotlib import rc_file
from matplotlib import pyplot as plt
rc_file('~/guiaraujo_medium.mplstyle')

def parse_arquivo_treinamento(caminho_arquivo):
    """
    Lê o arquivo e extrai o número de clientes, as épocas locais 
    e a lista de métricas de acurácia.
    """
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        conteudo = f.read()
        
    # 1. Número de clientes: Procura por um número seguido da palavra "clients"
    match_clients = re.search(r'(\d+)\s+clients', conteudo, re.IGNORECASE)
    num_clientes = int(match_clients.group(1)) if match_clients else None
    
    # 2. Número de épocas locais (local-epochs): Procura pelo padrão "local-epochs = [número]"
    match_epochs = re.search(r'local-epochs\s*=\s*(\d+)', conteudo)
    local_epochs = int(match_epochs.group(1)) if match_epochs else None
    
    # 3. Lista de acurácias: Procura todos os valores associados à chave 'accuracy'
    # O padrão busca "'accuracy': 'valor'" e extrai apenas o valor.
    matches_accuracy = re.findall(r"'accuracy':\s*'([^']+)'", conteudo)
    
    # Converte os valores extraídos (em notação científica) para ponto flutuante (float)
    lista_acuracia = [float(valor) for valor in matches_accuracy]
    
    # Monta e retorna o dicionário com os resultados
    return lista_acuracia

def plotar_grafico_acuracia(lista_dicionarios, titulo, nome_arquivo):
    # Cria uma nova figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Itera sobre cada dicionário na lista para criar as curvas
    for dados in lista_dicionarios:
        acuracias = dados
        
        # Converte as acurácias para porcentagem
        acuracias_pct = [acc * 100 for acc in acuracias]
        
        # O eixo x (Rodada global) pode ser o próprio índice da lista (0, 1, 2, ...)
        rodadas = list(range(len(acuracias)))

        # Adiciona a curva no gráfico
        ax.plot(rodadas, acuracias_pct, marker='o')
        
    # Configurações de exibição do gráfico
    ax.set_title(titulo)
    ax.set_xlabel("Rodada global")
    ax.set_ylabel("Acurácia (%)")
    
    # Adiciona a grade para facilitar a leitura
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Salva o gráfico no arquivo especificado e ajusta as margens
    plt.savefig(nome_arquivo, bbox_inches='tight')
    
    # Fecha a figura para liberar memória
    plt.close(fig)

if __name__ == "__main__":
    resultados = []
    for i in range(9,13):
        caminho_arquivo = f'{i}.txt'
        resultado = parse_arquivo_treinamento(caminho_arquivo)
        print(f"Resultados para {caminho_arquivo}:")
        print(resultado)
        resultados.append(resultado)
        print("\n")
    plotar_grafico_acuracia(resultados, "QCNN CQC com CIFAR10", "cqc_multiple.png")
