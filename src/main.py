import argparse

from model.model import Decompiler


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-gui", "--gui", help="If false, use command line interface, else use graphical user interface")
        
    args = parser.parse_args()
    model = Decompiler()
    if args.gui in ['F', 'f', 'False', 'false']:
        from ui.cli import CLI
        view = CLI(model)

    else:
        from ui.gui import GUI
        view = GUI(model)

    view.run()

if __name__ == '__main__':
    main()
