class File:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        print(f'Opening the file {self.filename}.')
        self.__file = open(self.filename, self.mode)
        return self.__file

    def __exit__(self, exc_type, exc_value, exc_traceback):
        print(f'Closing the file {self.filename}.')
        if not self.__file.closed:
            self.__file.close()

        return False


with File('data.txt', 'r') as f:
    print(int(next(f)))