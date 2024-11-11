import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('llm4dfm')

class MyClass:
    def my_method(self):
        return "Hello World"


def main():
    x = MyClass().my_method()
    print(x)

logger.info("llm4dfm loaded")
