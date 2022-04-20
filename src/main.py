from gui.view import View
from model.model import Decompiler


def main():
    model = Decompiler()
    view = View(model)

    view.run()

if __name__ == '__main__':
    main()
