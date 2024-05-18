class TestClass:
    def __init__(self, t, d1, d2):
        self.d1 = d1
        self.d2 = d2
        if t == 1:
            self.f = self.f1
        else:
            self.f = self.f2
        return

    def f1(self):
        print("1:", self.d1)

    def f2(self):
        print("2:", self.d2)



c1 = TestClass(1,5,7)

c2 = TestClass(2,8,19)

c1.f()

c2.f()