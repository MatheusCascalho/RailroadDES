# Máquina de estados

## Contexto

**Que tipo de problemas essa abordagem busca resolver?**
* Falta de clareza sobre como representar o sistema. A abordagem de otimização proposta inicialmente assume que o 
Roteador (classe Router) é um agente que conhece o estado do sistema e a partir desse estado define uma ação. Uma vez
a ação é tomada o ambiente (classe Railroad) retorna para o agente a recompensa relacionada a essa ação.
  * Quanto mais entidade temos na ferrovia, mais complexo é representar o estado desse sistema. A abordagem de implementar 
uma máquina de estados busca definir de forma clara como um estado é especificado e qual a sua representação para que o 
agente possa armazená-lo em memória.
  * Nem toda ação do agente é factível, assim como nem todo elemento de simulação. Por esse motivo é importante definir
de forma clara como um sistema sair de um estado para outro.

**Responsabilidade**: controlar a evolução do sistema por meio das transições de estado, bem como identificar e representar o 
estadoa atual do sistema.


## Descrição
Uma máquina de estados é um modelo computacional utilizado para representar sistemas que podem estar em um número finito de estados. Ela descreve o comportamento de um sistema através de um conjunto de estados, transições entre esses estados e ações associadas às transições ou aos próprios estados.

Em uma máquina de estados, o sistema sempre se encontra em um estado específico, e as mudanças de estado (ou transições) acontecem em resposta a eventos ou condições específicas. Cada transição leva o sistema de um estado para outro, com base em entradas ou eventos.

Aqui estão os principais componentes de uma máquina de estados:

* **Estados**: São as diferentes condições ou situações possíveis do sistema. Um estado é dito marcado se representar
o estado atual do sistema. 

* **Transições**: São as mudanças entre estados, geralmente causadas por eventos ou condições específicas.

* **Eventos**: São as entradas que causam as transições de estado. No caso de transições automáticas, um evento ocorre 
quando determinado estado observado pela transição passa a estar marcado.

* **Ações**: São atividades realizadas durante a transição ou  enquanto o sistema está em um estado específico.

## Propriedades

* Toda máquina deve possuir um, e apenas um, estado ativo;
* Um evento só pode habilitar uma transição por vez;
* Somente as transições do estado atual podem ser habilitadas;
* Uma máquina é dita bloqueante se algum estado não possuir transições;
* 