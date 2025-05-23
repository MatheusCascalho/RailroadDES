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
