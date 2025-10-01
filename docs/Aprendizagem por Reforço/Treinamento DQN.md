# Documentação do Treinamento DQNRouter

Este documento descreve a arquitetura e o funcionamento do **sistema de treinamento do DQNRouter**, implementado no arquivo `train_DQNRouter.py`.

---

## 📌 Visão Geral

O código implementa uma arquitetura **Actor-Learner distribuída**, inspirada em algoritmos como **Ape-X DQN**.  
A ideia central é separar o processo de **coleta de experiências** (atores) do processo de **aprendizado da rede neural** (learner).  

### Componentes principais:
- **Actor (Atores):** executam episódios da simulação, interagem com o ambiente e produzem experiências.
- **Learner (Aprendiz):** atualiza a rede neural (policy e target networks) com base nas experiências recebidas.
- **Logger:** registra métricas de desempenho de cada episódio.
- **Fila de Experiências:** conecta atores ao learner.
- **Fila de Saída:** conecta atores ao logger.

---

## ⚙️ Fluxo do Treinamento

1. **Carregamento do modelo base**
   - O estado inicial do sistema ferroviário é carregado de um arquivo `.dill`.
   - É construído o espaço de **estados** e o espaço de **ações**.

2. **Execução de Episódios (Actor)**
   - Cada ator roda uma simulação de 30 dias:
     - Observa o estado.
     - Escolhe uma ação (via DQN ou aleatória, conforme política).
     - Recebe a recompensa e transição.
   - As transições são enviadas para o **Learner** via `experience_queue`.
   - Estatísticas (volume operado, demanda atendida) são enviadas para `output_queue`.

3. **Atualização da Rede (Learner)**
   - O **Learner** roda em processo separado.
   - Ele consome experiências da `experience_queue` e aplica:
     ```python
     learner.update(experience)
     ```
   - Periodicamente salva os pesos das redes (`policy_net` e `target_net`).

4. **Registro dos Resultados (Logger)**
   - O **Logger** consome dados da `output_queue`.
   - Gera arquivos `.log` com métricas por episódio.

5. **Loop Principal**
   - Executa `TRAINING_STEPS`.
   - Em cada passo:
     - Lança processos de **atores**.
     - Espera terminarem.
     - Solicita ao **Learner** salvar os modelos atualizados.
   - Ao final, todos os processos são encerrados.

---

## 🧩 Diagrama da Arquitetura

### Arquitetura Geral

```mermaid
flowchart TD
    A[Actor Processos de Episódios] -->|Experiências| B[Experience Queue]
    B --> C[Learner]
    C -->|Atualiza Redes (Policy/Target)| D[(Modelos Salvos)]

    A -->|Métricas| E[Output Queue]
    E --> F[Logger]
    F --> G[(Arquivos de Log)]
```

## Detalhe do Loop de Treinamento

```mermaid
sequenceDiagram
    
    participant Main
    participant Actor
    participant Learner
    participant Logger

    Main -> Actor: Inicia episódios
    Actor->>Learner: Envia experiências (s, a, r, s')
    Actor->>Logger: Envia métricas (volume, demanda)
    Learner->>Learner: Atualiza policy_net e target_net
    Main->>Learner: Solicita salvamento do modelo
    Logger->>Logger: Registra logs em arquivo


```

## 📂 Estrutura de Processos

* Main Process
  * Cria e coordena subprocessos.

* Actor Processes
  * Rodam simulações independentes.

* Learner Process
  * Consome experiências e treina a rede.

* Logger Process
  * Armazena logs dos episódios.

## 🔑 Benefícios da Arquitetura

* Separação clara de responsabilidades:

  * Atores = coleta de dados.
  * Learner = atualização de rede.

* Escalabilidade:

  * Fácil aumentar o número de atores (paralelismo).

* Persistência:

  * Modelos são salvos periodicamente.
  * Experimentos são registrados em logs.

## 📊 Resumo

Essa implementação do DQNRouter é um sistema distribuído de aprendizado por reforço com arquitetura Actor-Learner.
Os episódios são gerados em paralelo, experiências são enviadas para o Learner, e os resultados são logados para análise futura.