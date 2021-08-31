class Stencil(object):
    def __init__(self, n) -> None:
        super().__init__()
        self.n = n
        self.stencil_name = "Unknown"
        self.backend_name = "Unknown"

    def __str__(self) -> str:
        return f"{self.stencil_name}|{self.backend_name}"

    def activate():
        raise NotImplementedError()

    def deactivate():
        raise NotImplementedError()

    def run():
        raise NotImplementedError()


class Dummy_Stencil(Stencil):
    def __init__(self, n) -> None:
        super().__init__(n)

    def __str__(self) -> str:
        return "Dummy Stencil"

    def activate(self):
        pass

    def deactivate(self):
        pass

    def run(self):
        time.sleep(self.n/1000)
