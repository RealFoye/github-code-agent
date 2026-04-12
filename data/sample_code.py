def helper():
    print(1)

def foo():
    helper()
    print(2)

class MyClass:
    def method_a(self):
        helper()
