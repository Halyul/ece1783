import pathlib, yaml, math, shutil
from lib.config.config_object import ConfigObject

class Config:

    def __init__(self, path, clean_up=False):
        self.config_path = pathlib.Path.cwd().joinpath(path)
        self.config = None
        self.__read_config()
        self.debug = self.config['debug'] if 'debug' in self.config else False
        self.output_path = Paths(self.config['output_path'])
        self.params = Params(self.config['params'])
        self.video_params = Params(self.config['video_params'])
        self.statistics = Statistics(self.config['statistics'])
        self.decoder = Decoder(self.config['decoder'])

        self.output_path.main_folder = pathlib.Path.cwd().joinpath(self.output_path.main_folder)
        if clean_up:
            self.__clean_up()
        self.__create_output_path()

    def __clean_up(self):
        if self.output_path.main_folder.exists():
            shutil.rmtree(self.output_path.main_folder)
        
    def __create_output_path(self) -> None:
        self.output_path.main_folder.mkdir(parents=True, exist_ok=True)
        for key, value in self.output_path.config.items():
            if key == 'main_folder':
                continue
            elif key.endswith('_folder'):
                setattr(self.output_path, key, self.output_path.main_folder.joinpath(value))
                current_path = getattr(self.output_path, key)
                current_path.mkdir(parents=True, exist_ok=True)
            elif key.endswith('_file'):
                setattr(self.output_path, key, self.output_path.main_folder.joinpath(value))

    """
        Read the config file.
    """
    def __read_config(self) -> None:
        try:
            self.config = yaml.load(open(self.config_path, "r"), Loader=yaml.FullLoader)
            for key, value in self.config.items():
                setattr(self, key, value)
        except Exception as e:
            raise
        if 'output_path' not in self.config:
            raise Exception('Output not found in config.')
    
    """
        Get the output path.

        Parameters:
            key (str): The key of the output path.

        Returns:
            pathlib.Path: The output path.
    """
    def get_output_path(self, key: str) -> pathlib.Path:
        return self.config['output_path'][key]

class Paths(ConfigObject):
    def validate(self):
        return

class Params(ConfigObject):
    def validate(self):
        if not (0 <= self.qp <= (math.log2(self.i) + 7)):
            raise Exception('Invalid qp value.')
        return
    
class Decoder(ConfigObject):
    def validate(self):
        self.input_path = Paths(self.input_path)
        self.output_path = Paths(self.output_path)

        self.input_path.mv_folder = pathlib.Path.cwd().joinpath(self.input_path.mv_folder)
        self.input_path.residual_folder = pathlib.Path.cwd().joinpath(self.input_path.residual_folder)
        self.input_path.meta_file = pathlib.Path.cwd().joinpath(self.input_path.meta_file)

        self.output_path.main_folder = pathlib.Path.cwd().joinpath(self.output_path.main_folder)

        if not self.output_path.main_folder.exists():
            self.output_path.main_folder.mkdir()
        return
    
class Statistics(ConfigObject):
    def validate(self):
        self.path = pathlib.Path.cwd().joinpath(self.path)
        if not self.path.exists():
            self.path.mkdir()
        return