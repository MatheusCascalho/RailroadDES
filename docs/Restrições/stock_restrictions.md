# Restrições de estoque

Uma importante restrição é a restrição de estoque, implementada pela classe `StockConstraint` e suas classes filhas.
O sistema é baseado no padrão de projeto _Observer_, onde as classes de restrição observam o estado dos estoques e agem com base nas condições de carga e espaço disponível.

## Estrutura de Classes

### `StockConstraint`
A classe `StockConstraint` é uma implementação de restrição baseada no estoque. Ela herda de `ProcessConstraintSystem` e implementa a interface `AbstractObserver` para notificar mudanças em objetos do tipo `StockInterface`.

#### Métodos:
- `__init__(self, train_size: float)`: Construtor que define o tamanho do trem (capacidade do estoque).
- `append_subject(self, sub)`: Adiciona um objeto do tipo `StockInterface` à lista de observadores. Chama o `update()` para garantir que o estado esteja sincronizado.
- `update(self, *args)`: Método a ser implementado nas subclasses para atualizar o estado baseado nas condições do estoque (não implementado diretamente aqui).

### `StockLoadConstraint`
A classe `StockLoadConstraint` é uma subclasse de `StockConstraint` que monitora a carga no estoque. Ela bloqueia ou libera o estado do processo dependendo se a quantidade de carga no estoque é suficiente para o tamanho do trem.

#### Métodos:
- `__init__(self, train_size: float)`: Construtor que define o tamanho do trem.
- `update(self, *args)`: Atualiza o estado do processo. Se o volume de carga for menor que o tamanho do trem, o estado é "bloqueado". Caso contrário, o estado é "liberado".

### `StockUnloadConstraint`
A classe `StockUnloadConstraint` também é uma subclasse de `StockConstraint`, mas monitora o espaço disponível no estoque para descarregar. O comportamento é semelhante ao de `StockLoadConstraint`, mas verifica o espaço livre ao invés da carga.

#### Métodos:
- `__init__(self, train_size: float)`: Construtor que define o tamanho do trem.
- `update(self, *args)`: Atualiza o estado do processo. Se o espaço disponível for menor que o tamanho do trem, o estado é "bloqueado". Caso contrário, o estado é "liberado".


## Fluxo de Funcionamento

1. **Criação da Máquina de Estados**: O `ProcessConstraintSystem` cria uma máquina de estados com três estados principais: "READY", "BUSY" e "BLOCKED". 
   - As transições são definidas para passar de "READY" para "BUSY" (iniciar processo) e de "BUSY" para "READY" (finalizar processo). 
   - Também são definidas transições para bloquear o processo ("READY" para "BLOCKED") e liberar o processo ("BLOCKED" para "READY").

2. **Aplicação das Restrições**: As classes `StockLoadConstraint` e `StockUnloadConstraint` observam os estoques (implementados como `StockInterface`). Elas verificam se a carga ou o espaço disponível são suficientes para o tamanho do trem.
   - Quando as condições de carga ou espaço não são atendidas, o processo é bloqueado.
   - Caso contrário, o processo é liberado e a transição do estado ocorre.

3. **Notificação de Mudanças**: O mecanismo de notificação é baseado no padrão Observer. O estoque (implementando `StockInterface`) notifica suas restrições sempre que o volume ou espaço muda. As restrições, por sua vez, atualizam o estado do processo.

## Exemplo de Uso

```python
from models.stock import OwnStock
from models.clock import Clock
from datetime import datetime, timedelta

# Criação do relógio e estoque
clk = Clock(start=datetime(2025, 4, 1), discretization=timedelta(hours=1))
stock1 = OwnStock(clock=clk, capacity=50, product='aço')

# Criação das restrições
restriction_1 = StockLoadConstraint(train_size=10)
restriction_2 = StockLoadConstraint(train_size=5)

# Adicionando observadores
stock1.add_observers([restriction_1, restriction_2])

# Verificando o estado
assert restriction_1.is_blocked()  # Bloqueado inicialmente
assert restriction_2.is_blocked()  # Bloqueado inicialmente

# Modificando o volume do estoque
stock1.receive(volume=15)

# Verificando novamente os estados após alteração
assert not restriction_1.is_blocked()  # Não bloqueado após receber carga
assert not restriction_2.is_blocked()  # Não bloqueado após receber carga

# Despachando carga
stock1.dispatch(volume=10)

# Verificando os estados após despacho
assert restriction_1.is_blocked()  # Bloqueado novamente após despacho
assert not restriction_2.is_blocked()  # Não bloqueado

```

# Considerações Finais
Este sistema proporciona um controle robusto e flexível sobre o comportamento dos processos ferroviários,
garantindo que as operações respeitem as restrições de carga e espaço nos estoques. A arquitetura de máquina de estados 
e o padrão Observer garantem uma solução escalável e facilmente extensível.
