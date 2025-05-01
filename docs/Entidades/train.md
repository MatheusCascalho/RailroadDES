# Trem - Train

**Classe**: `Train`

**Responsabilidade**: Realizar a sequencia de eventos que permita aceitar o volume de cada demanda.

**Colaboradores**:

* [Task](../Artefatos/time_table.md): Define a tarefa que o trem deve realizar.
* [TimeTable](../Artefatos/time_table.md): Registra os tempos do trem em cada momento.
* [ActivitySystem](../Recursos/Sistemas/ActivitySystem.md): [sistema](../Recursos/Sistemas/index.md) que controla as atividades do trem
* [LoadSystem](../Recursos/Sistemas/LoadSystem.md): [sistema](../Recursos/Sistemas/index.md) que controla a carga do trem


## Sequencia de eventos: 
  1. Receber novo fluxo - Router.new_flow();
  1. Ir até o nó de carregamento - Train.arrive();
     2. Agendado pelo RailSegment.send()
  1. Iniciar Carregamento - Train.start_load()
     2. Agendado pelo Node.process
        3. Que por sua vez é acionado pelo Node.process_constraint (máquina de estados) quando a transição 
        start é acionada pela chegada de um trem 
  1. Finalizar Carregamento - Train.finish_load()
     2. Agendado pelo Train.start_load()
  3. Ir para a fila de saída - Node.maneuver_to_dispatch()
     4. Agendado pelo Train.finish_load()
  1. Saír do nó de carregamento - Train.leave()
     2. Agendado pelo Node.dispatch()
  1. Ir até o nó de descarregamento - Train.arrive();
      2. Agendado pelo RailSegment.send()
  1. Descarregar - Train.unload()
     2. Agendado pelo Node.process

## Parâmetros
  1. Capacity - para load()
  1. Tempos de trânsito - para arrive()
     1. Fornecidos por RailSegment.get_transit_time()
  1. Path - para arrive()
     1. Fornecido por Railroad.get_path()

# Componentes

1. StateMachine - Responsável por Habilitar e disparar as transições de estado
2. TimeTable - Responsável por armazenar o histórico de atividades do trem
3. Task - Responsável por detalhar a tarefa atual do trem

## Callbacks
  1. Train.arrive() (após Train.leave())
  2. Train.leave() 
  2. Node.finish_load()/finish_unload()

## Propiedades
1. Um trem só pode descarregar se estiver na sua estação de destino;
2. Um trem só pode carregar se estiver na sua estação de origem;
3. Um trem só pode chegar em um lugar após ter saído do anterior;
4. Um trem deve registrar o instante de cada evento que faz com que ele evolua;
5. Um trem só pode ter seu estado atualizado se as condições de carga ou atividade forem modificadas;

