# Dual Stream Setup
class DualStream:
    def __init__(self, terminal_stream, file_stream):
        self.terminal_stream = terminal_stream
        self.file_stream = file_stream

    def write(self, message):
        self.terminal_stream.write(message)
        self.file_stream.write(message)

    def flush(self):
        # This flush method is needed for compatibility with certain functionalities like tqdm
        self.terminal_stream.flush()
        self.file_stream.flush()
