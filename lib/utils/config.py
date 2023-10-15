import pathlib, yaml

class Config:

    def __init__(self, path):
        self.config_path = pathlib.Path.cwd().joinpath(path)
        self.config = None
        self.__read_config()

    """
        Read the config file.
    """
    def __read_config(self) -> None:
        try:
            self.config = yaml.load(open(self.config_path, "r"), Loader=yaml.FullLoader)
        except Exception as e:
            raise
        if 'output_path' not in self.config:
            raise Exception('Output not found in config.')
    
    """
        Get the output path.

        Returns:
            pathlib.Path: The output path.
    """
    def get_output_path(self, key) -> pathlib.Path:
        return self.config['output_path'][key]
