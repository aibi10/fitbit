class EncryptionException(Exception,BaseException):
    def __init__(self,message):
        self.message=message

    def __repr__(self):
        return "EncryptionException"

    def message(self):
        return self.message()

    def __str__(self):
        return self.message
