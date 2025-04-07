# Train

**Responsabilidade**: Realizar a sequencia de eventos que permita aceitar o volume de cada demanda.

- Sequencia de eventos: 
  1. Receber novo fluxo - Router.new_flow();
  1. Ir até o nó de carregamento - Train.arrive();
  1. Carregar - Train.load()
  1. Saír do nó de carregamento - Train.leave()
  1. Ir até o nó de descarregamento - Train.arrive();
  1. Descarregar - Train.unload()

- Parâmetros
  1. Capacity - para load()
  1. Tempos de trânsito - para arrive()
     1. Fornecidos por RailSegment.get_transit_time()
  1. Path - para arrive()
     1. Fornecido por Railroad.get_path()

- Callbacks
  1. Train.arrive() (após Train.leave())
  2. Node.finish_load()/finish_unload()



