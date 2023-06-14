import logging

# Define your custom levels
TIMES = 21
STEPS = 22
DATA = 23

logging.addLevelName(TIMES, "TIMES")
logging.addLevelName(STEPS, "STEPS")
logging.addLevelName(DATA, "DATA")

def times(self, message, *args, **kws):
    if self.isEnabledFor(TIMES):
        self._log(TIMES, message, args, **kws)

def steps(self, message, *args, **kws):
    if self.isEnabledFor(STEPS):
        self._log(STEPS, message, args, **kws)

def data(self, message, *args, **kws):
    if self.isEnabledFor(DATA):
        self._log(DATA, message, args, **kws)

logging.Logger.times = times
logging.Logger.steps = steps
logging.Logger.data = data

# Define a custom handler that filters messages based on their level
class CustomHandler(logging.Handler):
    def __init__(self, levels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.levels = levels

    def emit(self, record):
        if record.levelno in self.levels:
            log_entry = self.format(record)
            print(log_entry)

# function to create a logger and add the custom handler to it
def create_logger(name, levels):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = CustomHandler(levels=levels)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    return logger

# logger = create_logger([TIMES, STEPS, DATA])
