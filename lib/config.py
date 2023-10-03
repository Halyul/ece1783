import pathlib, yaml

class Config:

    def __init__(self, path):
        self.config_path = pathlib.Path.cwd().joinpath(path)
        self.config = None
        self.__read_config()

    def __read_config(self):
        try:
            self.config = yaml.load(open(self.config_path, "r"), Loader=yaml.FullLoader)
        except Exception as e:
            raise