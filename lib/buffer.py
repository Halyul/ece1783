class Buffer:

    def __init__(self, RCSaver):
        self.buffer = {}
        self.RCSaver = RCSaver

    def add(self, func_name, key, value):
        if self.RCSaver:
            if func_name not in self.buffer:
                self.buffer[func_name] = {}
            self.buffer[func_name][key] = value

    def get(self, func_name, key):
        return self.buffer[func_name][key]

    def state(self, func_name, key):
        if self.RCSaver:
            if func_name in self.buffer:
                if key in self.buffer[func_name]:
                    return True
        return False
    
    def concatenate(self, buffer_to_add):
        for func_name in buffer_to_add.buffer:
            if func_name not in self.buffer:
                self.buffer[func_name] = {}
            for key in buffer_to_add.buffer[func_name]:
                self.buffer[func_name][key] = buffer_to_add.buffer[func_name][key]