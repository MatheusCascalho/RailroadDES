class Path:
    def __init__(self, path, is_moving: bool = True):
        self.starts_moving = is_moving
        self.path = path

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, new_path):
        new_path = self.path_aumentado(path=new_path, starts_moving=self.starts_moving)
        self.__path = new_path

    @staticmethod
    def path_aumentado(path, starts_moving):
        if not path:
            return path
        resultado = []
        for i in range(len(path) - 1):
            resultado.append(path[i])
            resultado.append(f"{path[i]}-{path[i + 1]}")
        resultado.append(path[-1])  # Adiciona o último item da lista
        if starts_moving:
            resultado.insert(0, f'_-{resultado[0]}')
        return resultado

    def walk(self):
        self.__path = self.__path[1:]

    @property
    def current_location(self):
        if len(self.__path)==0:
            return '' # Means that train finish travel, therefore, is not in load point
        return self.__path[0]

    def next_location(self):
        if self.__path[1] in self.__path[0]:
            return self.__path[1]
        return self.__path[2]
