'''
    Populacao: uma populacao é composta por um conjunto de individuos | populacao = cojunto de vetores
    Individuo: um individuo é composto por um determinado numero de genes | individuo = um vetor 
    Gene: elementos do vetor (pontos (x,y))
    Pais: sao considerados os melhores vetores/individuos (fitness score) para a geracao de fihos
    Filho: herda as caracteristicas dos pais

    Fitness: é uma funcao que avalia a melhor rota, ou seja, a rota com menor distancia
    Selecao: é uma funcao que seleciona os melhores pais/individuos de acordo com probabilidade
    Crossover: é uma funcao que gera filhos atraves dos pais selecionados anteriormente
    Mutacao: é uma funcao que faz a troca de um elemento do vetor de forma aleatoria  
    Elitismo: seleciona os melhores individuos da populacao
'''

import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(5)

# ---------------------------------------------------------------------------------------------------------------------------------
# Gera a Populacao Inicial (conjunto de individuos)
def populacao_inicial(num_genes, tamanho_populacao):
    populacao = []

    for _ in range(tamanho_populacao):
        # permutacao aleatoria para criar um individuo (cada um possuindo um determinado numero de genes)
        individuo = np.random.permutation(num_genes)    
        # gera o vetor da populacao inicial (composta pelos individuos criados)
        populacao.append(individuo)                     
    return np.array(populacao)

# ---------------------------------------------------------------------------------------------------------------------------------
# Faz o calculo da distancia euclidiana e calcula a distancia total
def distancia_total(individuo, genes):
    return sum(np.linalg.norm(genes[individuo[i]] - genes[individuo[i - 1]]) for i in range(len(individuo)))

# ---------------------------------------------------------------------------------------------------------------------------------
# Faz o processo de selecao (seleciona os melhores pais/individuos) de acordo com o fitness score 
def selecao(populacao, genes):
    # cria um vetor com os fitness score de cada individuo 
    fitness = np.array([1 / distancia_total(individuo, genes) for individuo in populacao])   

    #calcula a probabilidade de cada fitness para a selecao
    probabilities = fitness / fitness.sum()

     # Faz a escolha de quais pais serao selecionados de acordo com a probabilidade (ainda há a chance de indivíduos com fitness menor serem escolhidos)
    selecionados = populacao[np.random.choice(len(populacao), size=len(populacao), p=probabilities)]
    
    # Encontra o melhor individuo da populacao
    melhor_indice = np.argmax(fitness)
    melhor_pai = populacao[melhor_indice]

    
    return selecionados, melhor_indice, melhor_pai

# ---------------------------------------------------------------------------------------------------------------------------------
# O crossover faz a mintura das informacoes de dois vetores que foram selecionados (selecao)
def crossover(pai1, pai2):
    size = len(pai1)
    # seleciona aleatoriamente posicoes do pai1 que serao herdadas no filho (mistura de vetores)
    start, end = sorted(np.random.choice(range(size), size=2, replace=False))

    # preenche o vetor filho com -1 usando np.full para garantir que seja um array NumPy
    filho = np.full(size, -1)

    # preenche os elementos selecionados do pai1 no filho
    filho[start:end] = pai1[start:end]  

    # preenche o resto das posicoes nao alteradas com os valores do pai2
    pos = end % size
    for item in pai2:
        if item not in filho:
            filho[pos] = item
            pos = (pos + 1) % size
    
    return filho

# ---------------------------------------------------------------------------------------------------------------------------------
# Faz a mudanca de um gene do individuo de forma aleatoria 
def mutacao(individuo, taxa_mutacao):
    for i in range(len(individuo)):
        # gera um número aleatório entre 0 e 1. Se esse número for menor que a taxa de mutação, uma mutação ocorrerá naquele gene
        if np.random.rand() < taxa_mutacao:
            # faz a troca dos genes
            if i == len(individuo):
                individuo[i], individuo[i:i-1] = individuo[i:i-1], individuo[i]
            else:
                individuo[i], individuo[i:i+1] = individuo[i:i+1], individuo[i] 

# ---------------------------------------------------------------------------------------------------------------------------------
def elitismo(populacao, genes, num_elites):
    fitness = np.array([1 / distancia_total(individuo, genes) for individuo in populacao])
    elites = populacao[np.argsort(fitness)[:num_elites]]
    return elites

# ---------------------------------------------------------------------------------------------------------------------------------
# Funcao para calcular o o algoritmo genetico
def algoritmo_genetico(genes, tamanho_populacao, geracoes, taxa_crossover, taxa_mutacao, titulo, num_elites):
    num_genes = len(genes)
    populacao = populacao_inicial(num_genes, tamanho_populacao)

    melhor_solucao = None
    melhor_distancia = float('inf')
    distancias_por_geracao = []  # Lista para armazenar as melhores distâncias

    plt.figure(figsize=(10, 6))  # Define o tamanho da figura apenas uma vez
    
    # Para cada uma das geracoes, a função realiza um ciclo em busca de evoluir a população 
    for geracao in range(geracoes):
        populacao, melhor_indice, melhor_pai = selecao(populacao, genes)

        geracao_anterior = melhor_distancia
        populacao_sobrevivente = list(elitismo(populacao, genes, num_elites))  # Mantém 5 melhores
        
        # Para cada par de pais selecionados aleatoriamente, a função verifica se o cruzamento ocorrerá com base na taxa_crossover
        for _ in range(0, tamanho_populacao - num_elites, 2):  # Deixa espaço para os elites
             # Seleciona um pai aleatório, garantindo que o melhor pai seja sempre um deles
            pai2 = populacao[np.random.choice([i for i in range(len(populacao)) if i != melhor_indice])]
            
            if np.random.rand() < taxa_crossover:
                filho1 = crossover(melhor_pai, pai2)
                filho2 = crossover(pai2, melhor_pai)
            else:
                filho1, filho2 = melhor_pai, pai2
            
            # Para cada filho criado, eles passam pela função mutacao, onde pode ocorre a mutacao de acordo com a taxa_mutacao
            mutacao(filho1, taxa_mutacao)
            mutacao(filho2, taxa_mutacao)
            populacao_sobrevivente.extend([filho1, filho2])
        
        # Os filhos gerados são armazenados na lista populacao_sobrevivente que garante que a população não exceda o tamanho
        populacao = np.array(populacao_sobrevivente)[:tamanho_populacao]  

        # Avalia as melhores soluções e guarda os valores
        for individuo in populacao:
            distancia = distancia_total(individuo, genes)
            if distancia < melhor_distancia:
                melhor_distancia = distancia
                melhor_solucao = individuo

        distancias_por_geracao.append(melhor_distancia)  # Armazena a melhor distância

        if geracao_anterior == melhor_distancia:
            taxa_mutacao = min(taxa_mutacao + 0.05, 0.5)  # Limitando a taxa de mutacao

        if geracao % 10 == 0:
            # A cada geração, a função imprime a geração atual e a melhor distância encontrada até o momento
            print(f"Geracao {geracao + 1}: Melhor Distancia = {melhor_distancia}")
            plot_solucao(genes, melhor_solucao, titulo)

    plt.show()  # Mostra o gráfico final ao final da execução
    return melhor_solucao, melhor_distancia, distancias_por_geracao 

# ---------------------------------------------------------------------------------------------------------------------------------
# Funcao para medir o desempenho do algoritmo generico (tanto em termos de melhor solucao quanto em tempo de execucao)
def desempenho(genes, tamanho_populacao, geracoes, taxa_crossover, taxa_mutacao, titulo, num_elites):
    start_time = time.time()
    melhor_solucao, melhor_distancia, distancias_por_geracao = algoritmo_genetico(genes, tamanho_populacao, geracoes, taxa_crossover, taxa_mutacao, titulo, num_elites)
    end_time = time.time()
    
    print(f"Melhor Distancia: {melhor_distancia}, Tempo de Execucao: {end_time - start_time:.4f} segundos")
    
    plot_distancia_geracao(distancias_por_geracao) 

# ---------------------------------------------------------------------------------------------------------------------------------
# Faz a plotagem do grafico
def plot_solucao(genes, solucao, title):
    plt.clf()  # Limpa a figura

    # Plota os pontos
    plt.scatter(genes[:, 0], genes[:, 1], color='blue', marker='o')
    
    # Plota a rota
    rota = np.append(solucao, solucao[0])  # Cria um ciclo adicionando o primeiro ponto no final
    plt.plot(genes[rota, 0], genes[rota, 1], color='red', linestyle='-', linewidth=2)
    
    # Configurações do gráfico
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.axis('equal')  # Mantém a proporção
    plt.pause(0.1)  # Pausa para permitir a atualização do gráfico

# ---------------------------------------------------------------------------------------------------------------------------------
# Plota o gráfico da distância ao longo das gerações
def plot_distancia_geracao(distancias_por_geracao):
    # Plota o gráfico da distância ao longo das gerações
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(distancias_por_geracao)), distancias_por_geracao, color='blue', marker='o')
    plt.title('Evolução da Melhor Distância ao Longo das Gerações')
    plt.xlabel('Gerações')
    plt.ylabel('Melhor Distância')
    plt.grid()
    plt.show()
# ---------------------------------------------------------------------------------------------------------------------------------

# ==================================================================================================================================

# Valores a serem utilizados no algoritmo genetico
num_genes = 20
tamanho_populacao = 100
geracoes = 270
taxa_crossover = 0.8
taxa_mutacao=0.1
num_elites = 3

'''
# Numeros que deram certo para 10 genes
num_genes = 10
tamanho_populacao = 40
geracoes = 50
taxa_crossover = 0.8
taxa_mutacao=0.15
num_elites = 2

# Melhor distancia para 20 genes
num_genes = 20
tamanho_populacao = 100
geracoes = 270
taxa_crossover = 0.8
taxa_mutacao=0.1
num_elites = 3

# Melhor distância para 30 genes
num_genes = 30
tamanho_populacao = 700
geracoes = 450
taxa_crossover = 0.8
taxa_mutacao=0.1
num_elites = 5

# Melhor distância para 40 genes
num_genes = 40
tamanho_populacao = 2500
geracoes = 300
taxa_crossover = 0.8
taxa_mutacao=0.2
num_elites = 7

# Melhor distancia para 100 genes
num_genes = 100
tamanho_populacao = 700
geracoes = 3000
taxa_crossover = 0.8
taxa_mutacao=0.2
num_elites = 22

'''

# Geração de Pontos Uniformemente Distribuídos -----------------------------------------------------------------------------
genes_uni_distribuidos = np.random.rand(num_genes, 2)

# Mede o desempenho para os pontos uniformemente distribuídos
print("\nDesempenho para Pontos Uniformemente Distribuidos: ")
titulo = 'Solução para Pontos Uniformemente Distribuídos'
desempenho(genes_uni_distribuidos, tamanho_populacao, geracoes, taxa_crossover, taxa_mutacao, titulo, num_elites)

# Geração de Pontos em Círculo ---------------------------------------------------------------------------------------------
genes_circulo = np.array([(np.cos(angle), np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, num_genes, endpoint=False)])

# Mede o desempenho para os pontos em círculo
print("\nDesempenho para Pontos em Circulo: ")
titulo = 'Solução para Círculo de Pontos'
desempenho(genes_circulo, tamanho_populacao, geracoes, taxa_crossover, taxa_mutacao, titulo, num_elites)