# Sistema de controle dos processos

O objetivo desse sistema é informar e controlar quando um processo pode ser iniciado ou não.
O sistema está ocioso? Podemos começar um processamento agora? Essas são perguntas responsidas por esse sistema.

Esse sistema é composto por duas classes: `ProcessorSystem`, responsável por informar se a unidade de 
processamento está ocupada ou ociosa, e a classe `ProcessConstraintSystem`, responsável por controlar quando um processo
pode ser iniciado ou não em determinado nó (classe `Node`). 

## ProcessorSystem

```mermaid

stateDiagram-v2
    BUSY --> IDLE: finish_load/finish_unload
    IDLE --> BUSY: start_load/start_unload
```



## ProcessConstraintSystem

```mermaid

stateDiagram-v2
    READY --> PROCESSING: process
    PROCESSING --> READY: finish
    READY --> BLOCKED: block_to_process_train
    BLOCKED --> READY: release
```

