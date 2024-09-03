import sys
import os


def error_message_detail(error, error_detail: sys):

    # Here we get the info from error_detail parameter
    _, _, exc_tb = error_detail.exc_info()

    # Get the filename in which the current error occur
    filename = exc_tb.tb_frame.f_code.co_filename

    # Now , we have filename, line number and the error that occur
    error_message = "Error ocurred in file name [{0}] and the line number is [{1}] and error is [{2}]".format(
        filename, exc_tb.tb_lineno, str(error)
    )

    return error_message


class SensorCustomException(Exception):

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message
