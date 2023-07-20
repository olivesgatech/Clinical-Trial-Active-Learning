 ## config
class BaseConfig:
    def __init__(self, config: dict):
        self.__dict__ = config
        self.__build()

    def __build(self):
        for key, val in self.__dict__.items():
            if val == "None":
                val = None
                self.__dict__[key] = val
            if isinstance(val, dict):
                self.__dict__[key] = BaseConfig(val)
