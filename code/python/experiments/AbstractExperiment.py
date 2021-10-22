import logging
import sys


class AbstractExperiment:

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

        loggingFile = f"../../data/logs/log.{self.__class__.__name__}"
        self.fileh = logging.FileHandler(loggingFile, 'a')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s [%(module)s-%(funcName)s:%(lineno)d]- %(levelname)s - %(message)s')
        self.fileh.setFormatter(formatter)

        self.instantiateLogger()

    def instantiateLogger(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.fileh)
        #sys.stdout = StreamToLogger(self.logger, logging.DEBUG)
        #sys.stderr = StreamToLogger(self.logger, logging.ERROR)



class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


