# Observadores e Sujeitos

Um importante padrão de projeto utilizado nesse simulador é o padrão ["Observer"](https://refactoring.guru/design-patterns/observer).

Esse padrão é utilizado pois diversas transições de estado da simulação dependem do estado de outros objetos.
Por exemplo: Uma restrição de estoque só deve ser atualizada para o estado "blocked" quando o estoque for
insuficiente para carregar um trem. Nesse caso, a restrição "observa" o objeto estoque para assim atualizar
seu estado. A restrição é o **observador** e o estoque é o **sujeito** que está sendo observado.

A classe `AbstractSubject` é responsável por implementar o comportamento padrão de um sujeito qualquer, que nada mais é 
do que o método `.add_observers()`, responsável por fazer com que o observador conheça o sujeito e que o sujeito conheça
o observador.

Além disso, implementamos a classe `AbstractSubject` utilizando a metaclass `SubjectMetaDecorator`.
Essa metaclass é responsável fazer com que todos os métodos marcados com o decorator `@to_notify` notifiquem seus
respectivos observadores quando uma atividade específica for finalizada.

Por exemplo: sempre que um estoque executar a atividade de recebimento de volume (método `.receive()`), os observadores
do estoque serão notificados após o fim do recebimento. Se, por exemplo, esse recebimento fizer com que o volume do 
estoque seja suficiente para abastecer um trem, então a restrição de estoque deve ser atualizada para o estado `READY`.


