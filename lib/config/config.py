import pathlib, yaml, math, shutil
from lib.config.config_object import ConfigObject

class Config:

    def __init__(self, path, clean_up=False, config_override: dict={}):
        self.config_path = pathlib.Path.cwd().joinpath(path)
        self.config = None
        self.__read_config()
        self.__config_override(config_override)
        self.__serialize_config()
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
        """
            Clean up the output folder.
        """
        if self.output_path.main_folder.exists():
            shutil.rmtree(self.output_path.main_folder)

    def __config_override(self, override: dict) -> dict:
        for key, value in override.items():
            if key in self.config:
                if type(self.config[key]) == dict:
                    for k, v in value.items():
                        if k in self.config[key]:
                            self.config[key][k] = v
                        else:
                            raise Exception('Invalid config override key.')
                else:
                    self.config[key] = value
            else:
                raise Exception('Invalid config override key.')
        return self.config
    
    def __serialize_config(self) -> None:
        for key, value in self.config.items():
                setattr(self, key, value)
        return

    def __create_output_path(self) -> None:
        """
            Create the output path.
        """
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

    def __read_config(self) -> None:
        """
            Read the config file.
        """
        try:
            self.config = yaml.load(open(self.config_path, "r"), Loader=yaml.FullLoader)
        except Exception as e:
            raise
        if 'output_path' not in self.config:
            raise Exception('Output not found in config.')
    
    def get_output_path(self, key: str) -> pathlib.Path:
        """
            Get the output path.

            Parameters:
                key (str): The key of the output path.

            Returns:
                pathlib.Path: The output path.
        """
        return self.config['output_path'][key]

class Paths(ConfigObject):
    def validate(self):
        return

class Params(ConfigObject):
    def validate(self):
        """
            Validate the param ranges.
        """
        if 'qp' in self.config and not (0 <= self.qp <= (math.log2(self.i) + 7)):
            raise Exception('Invalid qp value.')
        if 'nRefFrames' in self.config and not (1 <= self.nRefFrames <= 4):
            raise Exception('Invalid nRefFrames value.')
        if 'VBSEnable' in self.config and self.i < 4:
            # enabled when the block size is power of 2
            n = self.i
            while (n != 1):
                    if (n % 2 != 0):
                        self.VBSEnable = False
                        print('VBS cannot be enabled when the block size is not power of 2.')
                    n = n // 2
            self.VBSEnable = True
        return
    
class Decoder(ConfigObject):
    def validate(self):
        """
            Create decoder paths.
        """
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
        """
            Create statistics paths.
        """
        self.path = pathlib.Path.cwd().joinpath(self.path)
        if not self.path.exists():
            self.path.mkdir()
        return