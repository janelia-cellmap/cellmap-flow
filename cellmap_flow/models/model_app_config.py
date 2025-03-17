from enum import Enum


class AppInput:
    pass

class TextInput(AppInput):
    def __init__(self, name, default_value=None, description=None):
        self.name = name
        self.default_value = default_value
        self.description = description

class SelectInput(AppInput):
    def __init__(self, name, options, default_value=None, description=None):
        self.name = name
        self.default_value = default_value
        self.options = options
        self.description = description

class ListInput(AppInput):
    def __init__(self, name, options, default_value=None, description=None):
        self.name = name
        self.options = options
        self.default_value = default_value
        self.description = description

class FileInput(AppInput):
    def __init__(self, name, default_value=None, description=None):
        self.name = name
        self.default_value = default_value
        self.description = description

class ModelDashboardConfig:
    pass


class MyModelConfig(ModelDashboardConfig):
    def __init__(self):
        self.elemns = [FileInput("")]
