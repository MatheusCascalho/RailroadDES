# Documentação - Sistema de Reabastecimento de Estoque

## Descrição Geral

Este sistema implementa um mecanismo de reabastecimento de estoque baseado em taxas de reposição para diferentes produtos e no tempo decorrido desde o último evento de recebimento. Ele utiliza um relógio (`Clock`) para calcular a quantidade de tempo entre eventos e reabastecer os estoques de acordo com as taxas pré-definidas para cada produto.

## Estrutura de Classes

### `StockReplenisherInterface`

Esta interface define o contrato para qualquer classe que deseje implementar um sistema de reabastecimento de estoque.

#### Métodos:
- `replenish(self, stock: list[StockInterface])`: Método abstrato que define o processo de reabastecimento. Deve ser implementado pelas classes que herdam essa interface.

### `ReplenishRate`

Esta classe de dados representa a taxa de reposição para um produto específico, juntamente com a discretização do tempo.

#### Atributos:
- `product (str)`: Nome do produto para o qual a taxa de reposição se aplica.
- `rate (float)`: A taxa de reposição, ou seja, o volume de produto a ser reabastecido por unidade de tempo.
- `discretization (timedelta)`: A discretização do tempo, que define o intervalo entre as atualizações. O padrão é 1 hora (`timedelta(hours=1)`).

### `SimpleStockReplanisher`

Classe que implementa a interface `StockReplenisherInterface`. Ela fornece um mecanismo simples para reabastecer estoques com base nas taxas de reposição associadas a cada produto.

#### Atributos:
- `replenish_rates (dict)`: Um dicionário que mapeia o nome do produto para a taxa de reposição (`ReplenishRate`) associada.
- `clock (Clock)`: Um objeto `Clock` usado para determinar o tempo decorrido entre os eventos de recebimento.

#### Métodos:
- `__init__(self, replenish_rates: list[ReplenishRate], clock: Clock)`: Construtor da classe. Inicializa as taxas de reposição e o relógio. O dicionário `replenish_rates` é populado com as taxas de reposição para cada produto.
- `replenish(self, stocks: list[StockInterface])`: Este método percorre a lista de estoques fornecida e realiza o reabastecimento para cada estoque baseado na taxa de reposição do seu produto. A quantidade de produto a ser reabastecida é calculada com base no tempo decorrido desde o último evento de recebimento. O volume a ser reabastecido nunca ultrapassa o espaço disponível no estoque.

### Fluxo de Funcionamento

1. **Definição das Taxas de Reposição**: O sistema aceita uma lista de taxas de reposição (`ReplenishRate`), onde cada taxa está associada a um produto e define a quantidade de produto a ser reabastecida por unidade de tempo.

2. **Cálculo do Reabastecimento**:
   - Para cada estoque, o método `replenish` tenta obter o último evento de recebimento utilizando o método `get_last_receive_event()`.
   - O tempo decorrido desde o último evento de recebimento é calculado com base no relógio (`Clock`).
   - O número de "passos" (intervalos de tempo) é calculado dividindo o tempo decorrido pela discretização definida na taxa de reposição.
   - A quantidade de produto a ser reabastecida é calculada multiplicando a taxa de reposição pelo número de passos, mas o volume não pode ultrapassar o espaço disponível no estoque.

3. **Atualização do Estoque**: O volume calculado é então adicionado ao estoque utilizando o método `receive()`.

### Exemplo de Uso

```python
from models.clock import Clock
from models.stock import OwnStock
from datetime import timedelta
from stock_replanish import SimpleStockReplanisher, ReplenishRate

# Configuração do relógio
clock = Clock(start=datetime(2025, 4, 1), discretization=timedelta(hours=1))

# Definindo as taxas de reposição
replenish_rates = [
    ReplenishRate(product='aço', rate=5.0),  # 5 unidades de aço por hora
    ReplenishRate(product='carvão', rate=3.0)  # 3 unidades de carvão por hora
]

# Criando o reabastecedor
replanisher = SimpleStockReplanisher(replenish_rates=replenish_rates, clock=clock)

# Criando um estoque de 'aço'
stock_aço = OwnStock(clock=clock, capacity=100, product='aço')

# Adicionando o estoque ao reabastecedor
replanisher.replenish([stock_aço])

# Verificando o volume após reabastecimento
print(stock_aço.volume)

```

# Considerações Finais
O sistema de reabastecimento de estoque é simples, mas eficiente para garantir que os estoques sejam atualizados periodicamente com base em taxas de reposição predefinidas. Ele permite que diferentes produtos tenham taxas de reposição independentes e ajusta o volume do estoque conforme o tempo decorrido.


