"""
Descrição Geral
Este módulo implementa um sistema de máquina de estados baseado no padrão de design State Machine. A máquina de estados gerencia um conjunto de estados e transições entre eles, permitindo a modelagem de sistemas que evoluem ao longo do tempo em resposta a eventos. O módulo permite que apenas um estado esteja marcado como ativo por vez, garantindo consistência no gerenciamento de estados e transições.

A máquina de estados pode ser configurada com observadores, que são notificados sempre que um estado é ativado ou desativado.

"""

from dataclasses import dataclass, field
from typing import Any, Callable

from models.observers import AbstractObserver, AbstractSubject, SubjectNotifier, to_notify


# @dataclass
class State(AbstractSubject):
    """
    Representa um estado dentro da máquina de estados. Cada estado tem um nome, uma flag is_marked para indicar se o
    estado está ativo e uma lista de observadores que serão notificados quando o estado mudar.
    """
    def __init__(self, name, is_marked: bool):
        self.name = name
        self.is_marked = is_marked
        super().__init__()

    def __str__(self):
        return str(self.name)

    __repr__ = __str__


    def __bool__(self):
        """
        Retorna True se o estado estiver marcado (ativo), caso contrário retorna False.
        :return:
        """
        return self.is_marked

    def __hash__(self):
        """
        Retorna o valor de hash do estado, baseado no seu nome.
        :return:
        """
        return hash(self.name)

    def __eq__(self, other):
        """
        Compara se o estado é igual a outro estado ou a uma string (nome de um estado).
        :param other:
        :return:
        """
        if isinstance(other, State):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False

    @to_notify()
    def activate(self):
        """
        Marca o estado como ativo, desmarcando quaisquer outros estados previamente ativos. Lança uma exceção se o
        estado já estiver ativo.
        :return:
        """
        if self.is_marked:
            raise Exception("Estado já está ativo!")
        self.is_marked = True

    @to_notify()
    def deactivate(self):
        """
        Desmarca o estado como ativo. Lança uma exceção se o estado não estiver ativo.
        :return:
        """
        if not self.is_marked:
            raise Exception("Estado não está ativo!")
        self.is_marked = False


@dataclass
class Transition(AbstractObserver):
    """
    Representa uma transição entre dois estados. Cada transição tem uma condição de origem, destino, ação e gatilho.
    Atributos:
        name (str): Nome da transição.

        origin (State): Estado de origem da transição.

        trigger (bool): Condição de gatilho da transição.

        action (Callable): Ação a ser executada durante a transição.

        destination (State): Estado de destino da transição.

        subjects (list): Lista de sujeitos (estados) observados pela transição.
    """
    name: str
    origin: State
    trigger: bool = field(init=False, default_factory=bool)
    destination: State
    action: Callable = field(default = lambda : None)
    subjects: list[State] = field(init=False, default_factory=list)

    def shoot(self):
        if self.trigger and self.origin.is_marked:
            self.origin.deactivate()
            self.destination.activate()
            self.trigger = False
        self.action()

    def update(self):
        """
        Atualiza o gatilho da transição com base no estado atual. Se o estado de origem estiver marcado, a transição
        será ativada e o estado de origem será desmarcado, enquanto o estado de destino será marcado.
        :param state:
        :return:
        """
        self.trigger = any(s.is_marked for s in self.subjects)
        self.shoot()

    def force_trigger(self):
        """
        Força a transição de estado, acionando o gatilho manualmente.
        :return:
        """
        self.trigger = True
        self.shoot()


class MultiCriteriaTransition(Transition):
    def update(self):
        self.trigger = all(s.is_marked for s in self.subjects)
        self.shoot()


class StateMachine:
    def __init__(
            self,
            transitions: list[Transition],
    ):
        """
        Representa a máquina de estados. Gerencia o conjunto de estados e transições, controlando o estado atual da máquina
        e permitindo a realização de transições.
        """
        self.machine = {}
        self.transitions = {
            t.name: t
            for t in transitions
        }
        self.states: dict[Any, State] = {}
        for transition in transitions:
            self.states[transition.origin.name] = transition.origin
            self.states[transition.destination.name] = transition.destination

            if transition.origin not in self.machine:
                self.machine[transition.origin] = [transition]
            else:
                self.machine[transition.origin].append(transition)
            if transition.destination not in self.machine:
                self.machine[transition.destination] = []

        self.__current_state = self.get_current_state()

    @property
    def current_state(self) -> State:
        """
         Getter para o estado atual da máquina. Retorna o estado ativo ou o estado marcado mais recente.
        :return:
        """
        if self.__current_state.is_marked:
            return self.__current_state
        else:
            self.__current_state = self.get_current_state()
            return self.__current_state

    @current_state.setter
    def current_state(self, value: State):
        """
        Setter para o estado atual da máquina. Marca o estado como ativo, se ele não estiver ativo.
        """
        if not value.is_marked:
            raise Exception("O estado deve estar marcado para ser atribuido como estado atual do sistema")
        self.__current_state = value

    def allowed_transitions(self) -> list[Transition]:
        """
        Retorna uma lista de transições permitidas a partir do estado atual.
        :return:
        """
        if not self.current_state.is_marked:
            self.current_state = self.get_current_state()
        return self.machine[self.current_state]

    def get_current_state(self) -> State:
        """
        Retorna o estado atual da máquina, verificando qual estado está marcado.
        :return:
        """
        for state in self.machine:
            if state.is_marked:
                return state
        raise Exception('Nenhum estado está ativado')

    def update(self, transition=None):
        if transition is None:
            allowed_transitions = self.allowed_transitions()
            allowed_transitions[0].force_trigger()
        else:
            if transition not in self.transitions:
                raise Exception("Transition doesn't exist!")
            self.transitions[transition].force_trigger()


