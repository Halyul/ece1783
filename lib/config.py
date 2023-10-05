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
    
    def get_output_path(self, key):
        return pathlib.Path.cwd().joinpath(self.config['output'][key]) if key in self.config['output'] else None