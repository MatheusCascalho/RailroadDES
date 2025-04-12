# TimeTable

## Objetivo
O módulo `TimeTable` é responsável por gerenciar os registros de tempo durante a simulação ferroviária. Ele contém classes e métodos que controlam eventos como chegada, início e fim do processo, e partida dos itens no processo ferroviário. O sistema permite calcular o tempo total de fila, tempo de trânsito e tempo de utilização.

---

## Classes

### 1. EventName
Enumeração que define os tipos de eventos possíveis no processo:

- **ARRIVE**: Representa o evento de chegada.
- **START_PROCESS**: Representa o evento de início do processo.
- **FINISH_PROCESS**: Representa o evento de término do processo.
- **DEPARTURE**: Representa o evento de partida.

### 2. TimeEvent
Classe que representa um evento de tempo no sistema. Cada evento é associado a um nome de evento e um instante específico.

- **Atributos**:
  - `event`: Tipo do evento, que pode ser um dos valores definidos em `EventName`.
  - `instant`: Data e hora em que o evento ocorre, do tipo `datetime`.

### 3. TimeRegister
Classe que representa o registro de tempo de um processo em um item. Para cada item, são registrados os tempos de chegada, início e término do processo e partida.

- **Atributos**:
  - `process`: Processo relacionado ao item, do tipo `Process` (importado de `models.constants`).
  - `arrive`: Data e hora da chegada.
  - `start_process`: Data e hora de início do processo.
  - `finish_process`: Data e hora de término do processo.
  - `departure`: Data e hora de partida.

- **Métodos**:
  - `get_queue()`: Retorna o tempo total de fila como um `timedelta`, calculado pela diferença entre o tempo de chegada e o início do processo e entre o término do processo e a partida.
  - `get_process_time()`: Retorna o tempo total do processo como um `timedelta`, calculado pela diferença entre o tempo de término e o de início do processo.
  - `is_initial_register`: Propriedade que indica se o registro é inicial (baseado na presença ou ausência dos tempos de chegada, início do processo e término).
  - `update(event: TimeEvent)`: Atualiza o registro de tempo com os dados do evento fornecido.



### 4. TimeTable
Classe principal que gerencia os registros de tempo (`TimeRegister`) durante a simulação. Cada evento de tempo é registrado e a classe calcula o tempo total de fila, tempo de trânsito e tempo de utilização.

- **Atributos**:
  - `registers`: Lista de objetos `TimeRegister` que representam os registros de tempo durante a simulação.

- **Métodos**:
  - `update(event: TimeEvent)`: Atualiza o último registro de tempo com o evento fornecido.
  - `queue_time`: Propriedade que retorna o tempo total de fila acumulado, calculado somando os tempos de fila de todos os registros.
  - `in_transit_time`: Propriedade que retorna o tempo total de trânsito acumulado, calculado pela diferença entre as partidas e chegadas consecutivas dos registros.
  - `util_time`: Propriedade que retorna o tempo total de utilização acumulado, que é a soma do tempo de trânsito e dos tempos de processo dos registros.

### Regras de Atualização

1. **Sequência de Eventos**:  
   A sequência dos eventos é **estritamente controlada**. O método garante que os eventos sigam a ordem correta: `ARRIVE`, `START_PROCESS`, `FINISH_PROCESS`, e `DEPARTURE`.  
   Se um evento for registrado fora dessa sequência, uma exceção será levantada.

2. **Validação da Sequência de Eventos**:  
   Quando um evento é registrado, o método verifica se todos os **eventos anteriores** na sequência já foram definidos (ou seja, seus timestamps não são `None`).  
   Se **todos os eventos anteriores** na sequência já foram definidos, o evento pode ser registrado. Caso contrário, uma exceção será levantada.

3. **Verificação da Inicialização do Registro**:  
   Se o registro for considerado "inicial" (onde o evento `DEPARTURE` está presente, mas outros eventos como `ARRIVE`, `START_PROCESS` e `FINISH_PROCESS` estão ausentes), o método valida que todos os eventos anteriores tenham sido registrados antes de permitir o registro do evento atual.

4. **Controle de Registro de Evento**:  
   Se um evento já tiver sido registrado para o tipo de evento específico, uma exceção será levantada. Isso evita a duplicação de eventos dentro de um único registro.

5. **Exceção para Desalinhamento de Sequência**:  
   Se um evento de **`DEPARTURE`** for registrado antes de **`START_PROCESS`** ou **`FINISH_PROCESS`**, ou se eventos anteriores não forem registrados na sequência, uma exceção específica será levantada, indicando que a sequência de eventos não foi respeitada.

6. **Método `__check_sequence`**:  
   Esse método auxilia na verificação se a sequência de eventos foi respeitada. Ele conta as mudanças no estado de cada evento na sequência e garante que o número de alterações seja válido. Se houver mais de uma mudança na sequência (por exemplo, tentar registrar eventos em uma ordem incorreta), o método retorna `True` e a exceção é levantada.

---

## Fluxo de Atualização e Cálculos

1. **Registro de Evento**: Quando um evento ocorre (ex: chegada, início do processo, etc.), ele é registrado no `TimeTable` através do método `update`. O evento é registrado no último `TimeRegister` da lista ou cria-se um novo registro se necessário.

2. **Cálculos**:
   - **Tempo de Fila**: Calculado pela diferença entre o início do processo e a chegada, e entre a partida e o término do processo.
   - **Tempo de Trânsito**: Calculado pela diferença entre as partidas e chegadas consecutivas.
   - **Tempo de Utilização**: Somatório do tempo de trânsito e dos tempos de processo.

---

## Exemplo de Uso

```python
# Criando um TimeTable vazio
time_table = TimeTable()

# Registrando eventos
event1 = TimeEvent(event=EventName.ARRIVE, instant=datetime(2025, 4, 10, 12, 0))
time_table.update(event1)

event2 = TimeEvent(event=EventName.START_PROCESS, instant=datetime(2025, 4, 10, 12, 15))
time_table.update(event2)

event3 = TimeEvent(event=EventName.FINISH_PROCESS, instant=datetime(2025, 4, 10, 12, 45))
time_table.update(event3)

event4 = TimeEvent(event=EventName.DEPARTURE, instant=datetime(2025, 4, 10, 13, 0))
time_table.update(event4)

# Acessando as propriedades
print(f"Tempo de Fila Total: {time_table.queue_time}")
print(f"Tempo de Trânsito Total: {time_table.in_transit_time}")
print(f"Tempo de Utilização Total: {time_table.util_time}")