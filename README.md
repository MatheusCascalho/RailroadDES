# RailroadDES
Simulador a eventos discretos de um sistema ferroviário

## Sobre o projeto

O RailroadDES é uma plataforma para modelar e analisar operações ferroviárias por meio de simulação a eventos discretos. O modelo representa trens transportando cargas entre pontos de carregamento e descarregamento, considerando trechos ferroviários, nós, filas, recursos operacionais, estoques e horários.

O simulador processa eventos como chegadas, início e fim de operações e liberações de trens. Um roteador define as tarefas e os fluxos atendidos pelos trens. Sobre essa estrutura, o projeto permite experimentar estratégias de roteamento e otimização, incluindo métodos baseados em soluções e abordagens de aprendizado por reforço, como Q-learning e DQN.

As simulações podem ser analisadas por métricas e gráficos de volume transportado, cronogramas dos trens e níveis de estoque. O objetivo é comparar estratégias e observar seus efeitos no atendimento da demanda e no uso dos recursos ferroviários.

## Requisitos

* Python version: 3.11
* SO: Ubuntu
