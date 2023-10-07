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
        if 'output' not in self.config:
            raise Exception('Output not found in config.')
    
    """
        Get the output path.

        Returns:
            pathlib.Path: The output path.
    """
    def get_output_path(self) -> pathlib.Path:
        if 'args' not in self.config['output']:
            raise Exception('Output args not found in config.')
        if 'path' not in self.config['output']['args']:
            raise Exception('Output args path not found in config.')
        return pathlib.Path.cwd().joinpath(self.config['output']['args']['path'])
    
    """
        Get the output function.

        Returns:
            str: The output function name.
    """
    def get_output_func(self) -> str:
        return self.config['output']['func'] if 'func' in self.config['output']else None