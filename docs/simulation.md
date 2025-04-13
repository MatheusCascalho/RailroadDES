# Simulação

## Máquinas de Estado

### Carga do trem - LoadState

Transições:

* start_loading/start_unloading: só é disparada se a atividade do trem for Processing
* finish_loading/finish_unloading: só é disparada quando a ativdade for queue_to_leave

```mermaid
stateDiagram-v2
    %% [*] --> Cheio
    EMPTY --> LOADING: start_loading
    LOADING --> LOADED: finish_load
    LOADED --> UNLOADING: start_unloading
    UNLOADING --> EMPTY: finish_unload
```

### Atividade do trem - Activity

```mermaid
stateDiagram-v2
    MOVING --> QUEUE_TO_ENTER: arrive
    QUEUE_TO_ENTER --> PROCESSING: start_load/start_unload
    PROCESSING --> QUEUE_TO_LEAVE: finish_load/finish_unload
    QUEUE_TO_LEAVE --> MOVING: leave
```

### Equipamento de processamento
```mermaid

stateDiagram-v2
    BUSY --> IDLE: finish_load/finish_unload
    IDLE --> BUSY: start_load/start_unload
```

### Habilitação de processamento

```mermaid

stateDiagram-v2
    READY --> PROCESSING: process
    PROCESSING --> READY: finish
    READY --> BLOCKED: block_to_process_train
    BLOCKED --> READY: release
```

### Atendimento de fluxo - TravelState

```mermaid
stateDiagram-v2
    FINISHED --> RUNNING: new_flow
    RUNNING --> INVOICED: finish_load
    INVOICED --> FINISHED: finish_unload
```

## Estados Compostos

- TRAIN N -> LOAD_STATE | ACTIVITY | TRAVEL_STATE
- NODE M -> PROCESSOR_STATE | QUEUE Q_ENTER | QUEUE Q_LEAVE
- RAIL_SEGMENT R - TO_ORIGIN | TO_DESTINATION

## Ações

-> NEW_TRAVEL 

## Entidades


```mermaid

classDiagram
    class Train {
        %% Atributos
        +int train_id 
        +float max_capacity
        +float volume
        +string next_station
        +string current_station
        +Flow flow
        +Activity activity
        +LoadState load_state
                
        +load(volume_a_adicionar: float, tempo_carregamento: float)
        +unload(volume_a_reduzir: float, tempo_descarregamento: float)
        +arrive(novo_local: string)
        +leave(nova_parada: string, tempo_transito: float)
    }

    class Flow {
        +string product
        +string origin
        +string destination 
        +TravelState state
    }

    class Activity {
        <<enumeration>>
        +MOVING
        +LOADING
        +UNLOADING
        +WAITING_ON_QUEUE_TO_ENTER
        +WAITING_ON_QUEUE_TO_LEAVE
    }
    
    class LoadState {
        <<enumeration>>
        +FULL
        +EMPTY
    }
    
    class TravelState {
        <<enumeration>>
        +FINISHED
        +RUNNING
        +INVOICED
    }
    

    Train "1" --> "1" Activity : activity
    Train "1" --> "1" Flow : fluxo_atual
    Train "1" --> "1" LoadState : load_state
%%    Train "1" --* "1" Railroad : fluxo_atual
    Flow "1" --> "1" TravelState : state


    class RailSegment {
        %% Atributos
        +string origin
        +string destination
        +list[Train] to_origin
        +list[Train] to_destination
        +timedelta time_to_origin
        +timedelta time_to_destination
        +get_snapshot()

    }
    Processor "1" *-- "1" Train : current_train
    Queue "1" *-- "1" Train : to_origin/to_destination
    RailSegment "1" *-- "1" Train : to_origin/to_destination


    class Node {
        %% Atributos
        +string name
        +Queue to_enter
        +Queue to_leave
        +list[Processor] processors
        +list[RailSegment] RailSegmentes

    }
    
        
    class Queue {
        +list[Train] queue
        
        +receive()
        +remove()
        +get_snapshot()
    }
    
    class Processor {
        +ProcessorState state
        +Train current_train
        
        +receive()         
        +get_snapshot()
    }
    class LoadProcessor{
        +finish_load()
    }
    class UnloadProcessor{
        +finish_unload()
    }
    Processor <|-- LoadProcessor : processors
    Processor <|-- UnloadProcessor : processors

    
    class ProcessorState {
        <<enumeration>>
        +BUSY
        +IDLE
    }
    Node "1" *-- "N" RailSegment : processors     
    Node "1" *-- "N" Processor : processors
    Processor "1" *-- "1" ProcessorState : state

    Node "1" *-- "2" Queue : to_enter/to_leave
    
    class Railroad{
        +list[Nodes] nodes
        +list[RailSegment] RailSegmentes
%%        +list[Train] finished_trains
        
%%        +simulate()
        +feed_router()
    }
    Railroad "1" *-- "N" RailSegment : RailSegmentes
    Railroad "1" *-- "N" Node : nodes

    class Simulator{
        +Router router
        +Railroad railroad
        +Calendar calendar
        +Clock clock
        +simulate()
    }

    class Router{
        +list[demands] demands
        +new_flow()
        +add_feedback()
    }
    class Calendar{
        +list[Events] events
        +update()
        +push()
        +pop()
    }
    class Clock{
        +datetime current_time
    }
    
    Simulator "1" *-- "1" Router : router
    Simulator "1" *-- "1" Railroad : railroad
    Simulator "1" *-- "1" Calendar : calendar
    Simulator "1" *-- "1" Clock : clock

```

## Responsabilidades

* **Railroad**: É o modelo completo. É responsável por conter todas as entidades menores e representar o estado global do sistema para o simulador
  * Implicação: caso o estado seja representado de forma diferente, uma classe irmã de railroad deve ser implementada

* **Router**: Responsável por injetar eventos de decisão - new_flow()
  * Implicação: Aqui temos o algoritmo de otimização. Se a estratégia de otimização mudar, uma classe irmã deve ser implementada

* **Calendar**: Responsável por armazenar, ordenar e dispara os eventos de cada entidade;

* **Node**: Responsável por implementar as restrições de consumo e liberação de trens;

* **RailSegment**: Responsável por representar a posição dos trens quando não estiverem parados em nós;

* **Queue**: Responsável por representar os trens em estado ocioso da ferrovia.