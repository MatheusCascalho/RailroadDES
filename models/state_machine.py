"""
Descrição Geral
Este módulo implementa um sistema de máquina de estados baseado no padrão de design State Machine. A máquina de estados gerencia um conjunto de estados e transições entre eles, permitindo a modelagem de sistemas que evoluem ao longo do tempo em resposta a eventos. O módulo permite que apenas um estado esteja marcado como ativo por vez, garantindo consistência no gerenciamento de estados e transições.

A máquina de estados pode ser configurada com observadores, que são notificados sempre que um estado é ativado ou desativado.

"""

from dataclasses import dataclass, field
from typing import Any, Callable, Union
from abc import ABCMeta, abstractmethod

class AbstractObserver:
    """
    Classe abstrata para implementações de observadores. Os observadores são notificados quando um estado é alterado.
    """
    subjects: list

    @abstractmethod
    def update(self, *args):
        """
        Método abstrato que deve ser implementado para lidar com as notificações de alteração de estado.
        :param args:
        :return:
        """
        pass


@dataclass
class State:
    """
    Representa um estado dentro da máquina de estados. Cada estado tem um nome, uma flag is_marked para indicar se o
    estado está ativo e uma lista de observadores que serão notificados quando o estado mudar.
    """
    name: Any
    is_marked: bool
    observers: list[AbstractObserver] = field(default_factory=list)

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

    def activate(self):
        """
        Marca o estado como ativo, desmarcando quaisquer outros estados previamente ativos. Lança uma exceção se o
        estado já estiver ativo.
        :return:
        """
        if self.is_marked:
            raise Exception("Estado já está ativo!")
        self.is_marked = True
        self.notify()

    def deactivate(self):
        """
        Desmarca o estado como ativo. Lança uma exceção se o estado não estiver ativo.
        :return:
        """
        if not self.is_marked:
            raise Exception("Estado não está ativo!")
        self.is_marked = False
        self.notify()

    def notify(self):
        """
        Notifica todos os observadores associados ao estado.
        :return:
        """
        for observer in self.observers:
            observer.update(self)

    def add_observer(self, observer: list[AbstractObserver]):
        """
        Adiciona um ou mais observadores ao estado.
        :param observer:
        :return:
        """
        if not isinstance(observer, list):
            observer = [observer]
        self.observers.extend(observer)
        for observer in observer:
            if self not in observer.subjects:
                observer.subjects.append(self)


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

    def update(self, state: State):
        """
        Atualiza o gatilho da transição com base no estado atual. Se o estado de origem estiver marcado, a transição
        será ativada e o estado de origem será desmarcado, enquanto o estado de destino será marcado.
        :param state:
        :return:
        """
        self.trigger = state.is_marked
        self.shoot()

    def force_trigger(self):
        """
        Força a transição de estado, acionando o gatilho manualmente.
        :return:
        """
        self.trigger = True
        if self.trigger and self.origin.is_marked:
            self.origin.deactivate()
            self.destination.activate()
            self.trigger = False


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
        for transition in transitions:
            if transition.origin not in self.machine:
                self.machine[transition.origin] = [transition]
            else:
                self.machine[transition.origin].append(transition)
            if transition.destination not in self.machine:
                self.machine[transition.destination] = []

        self.__current_state = self.get_current_state()

    @property
    def current_state(self):
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


