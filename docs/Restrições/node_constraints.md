# Sistema de Restrições Ferroviárias

## Descrição Geral

O arquivo `node_constraints.py` implementa um sistema de restrições sobre o comportamento dos processos ferroviários, 
utilizando uma máquina de estados para controlar a transição entre os estados de "pronto", "ocupado" e "bloqueado". 

A classe base para o funcionamento da restrição é a classe `ProcessConstraintSystem`, que implementa a construção da 
máquina de estados de uma restrição.

Todas as restrições da simulação devem herdar de `ProcessConstraintSystem` e especificar seu próprio comportamento de 
restrição, de forma que sempre que uma restrição seja ativada a máquina de estado seja atualizada para o estado `blocked`.


## Estrutura de Classes

### `ProcessConstraintSystem`
A classe `ProcessConstraintSystem` é a classe base que define o sistema de restrições do processo ferroviário, utilizando uma máquina de estados para gerenciar as transições entre os estados de um processo.

#### Métodos:
- `__init__(self)`: Construtor da classe, que inicializa o sistema de eventos discretos.
- `build_state_machine(self) -> StateMachine`: Constrói e retorna a máquina de estados, configurando as transições entre os estados de "ready", "BUSY" e "blocked".
- `is_blocked(self)`: Verifica se o estado atual do processo é "bloqueado".





