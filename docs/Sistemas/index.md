# Sistemas 

No contexto deste projeto, Sistemas são objetos (classe `DiscreteEventSystem`) que representam entidades regidas por
máquinas de estados. Sempre que estivermos interessados em representar de forma clara o estado de um sistema, 
bem como impor regras bem definidas de transição de estados e ações associadas a essas transições, implementaremos um 
objeto concreto que irá implementar a classe `DiscreteEventSystem`. Essa class define o método abstrato
`build_state_machine()`, responsável por construir a máquina de estados do sistema, além de implementar o método
`__repr__` que facilita o processo de depuração de código quando estivermos utilizando algum sistema.
