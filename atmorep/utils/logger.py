import logging
import pathlib
import os

class RelPathFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)
        self.root_path = pathlib.Path(__file__).parent.parent.parent.resolve()

    def format(self, record):
        # Replace the full pathname with the relative path
        record.pathname = os.path.relpath(record.pathname, self.root_path)
        return super().format(record)

# support modern style string formatting, see: https://stackoverflow.com/questions/75715376/set-the-formatting-style-for-logging-percent-format-style-bracket-or-f-stri/75715402#75715402
class BracketStyleRecord(logging.LogRecord):
  def getMessage(self):
    msg = str(self.msg)  # see logging cookbook
    if self.args:
      try:
        msg = msg % self.args  # retro-compability for 3rd party code
      except TypeError as e:
        if e.args and "not all arguments converted" in e.args[0]:
          # "format" style
          msg = msg.format(*self.args)
        else:
          raise  # other Errors, like type mismatch
    return msg


logging.setLogRecordFactory(BracketStyleRecord) # enable modern style formatting
logger = logging.getLogger('atmorep')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = RelPathFormatter('%(pathname)s:%(lineno)d : %(levelname)-8s : %(message)s')
ch.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(ch)
