```mermaid
classDiagram
    
    class Railroad
    
    class AbstractNodeFactory{
        +create_rates()
        +create_constraints()
        +create_process_system()
        +create_node()
    }
    
    class TerminalFactory{
    }
    
    class ThirdTerminalFactory
    
    class Node
    
    class Terminal
    
    class ThirdTerminal
    
    %% Relação entre objetos
    Railroad --> AbstractNodeFactory
    Railroad --> Node
    Node <|-- Terminal
    Node <|-- ThirdTerminal
    
    AbstractNodeFactory <|-- TerminalFactory
    AbstractNodeFactory <|-- ThirdTerminalFactory
    TerminalFactory ..> Terminal : 
    ThirdTerminalFactory ..> ThirdTerminal : 
    



```