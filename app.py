from Sign_Language_Recognition.logger import logging
from Sign_Language_Recognition.exception import SignException
import sys


#logging.info("Welcome to the project")

try: 
    a = 7 / '6'
except Exception as e:
    raise SignException(e, sys) from e
    # logging.error("Error occurred in the project")
    # logging.error(e, sys)
    # logging.error(SignException(e, sys))
    # logging.error(SignException(e, sys).error_message)
