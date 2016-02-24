

class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
    expression -- input expression in which the error occurred
    message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

    def __repr__(self):
        return 'InputError(%r, %r)' % (self.expression, self.message)

    def __str__(self):
        return 'Expression: %s\nMessage: %s' % (self.expression, self.message)
