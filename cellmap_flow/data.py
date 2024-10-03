import yaml
from typing import List, Dict


class Checkpoint:
    def __init__(self, number: int, path: str):
        self.number = number
        self.path = path

    def to_dict(self):
        return {"number": self.number, "path": self.path}

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Iteration : {self.number}"

    @staticmethod
    def from_dict(data: dict):
        return Checkpoint(number=data["number"], path=data["path"])


class Model:
    def __init__(self, name: str, checkpoints: List[Checkpoint]):
        self.name = name
        self.checkpoints = checkpoints

    def to_dict(self):
        return {
            "name": self.name,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
        }

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Model(name={self.name}, checkpoints={self.checkpoints})"

    @staticmethod
    def from_dict(data: dict):
        return Model(
            name=data["name"],
            checkpoints=[Checkpoint.from_dict(cp) for cp in data["checkpoints"]],
        )


def generate_from_yaml(path: str) -> List[Model]:
    with open(path, "r") as file:
        data = yaml.safe_load(file)["models"] 
        models = [Model.from_dict(model) for model in data]
        return {model.name: model for model in models}


def serialize_to_yaml(models: List[Model], path: str):
    with open(path, "w") as file:
        data = {"models": [model.to_dict() for model in models]} 
        yaml.safe_dump(data, file)
