# Task

**Responsabilidade**: registrar a decisão do roteador associada um trem em determinado momento da simulação, bem como o efeito dessa decisão

**Justificativa**: esse objeto será importante para avaliar o impacto de cada decisão, refinando o algoritmo de aprendizado de máquina

## Components

* TimeTable
* Demand

## Métodos

* update(): atualiza a timetable da task
* penalty(): obtém a penalização da task (exemplo: fila associada a decisão)
* reward(): obtém a recompensa da decisão tomada (exemplo: volume operado na decisão)
