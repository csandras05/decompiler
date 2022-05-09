import argparse
import os

from model import utils
from model.model import Decompiler

CUR_DIR = os.path.dirname(__file__)


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-gui", "--gui", help="If false, use command line interface, else use graphical user interface")
    args = parser.parse_args()
    
    config = utils.load_yaml(f'{CUR_DIR}/model/model_config.yaml')
    
    model = Decompiler(config)
    if args.gui in ['F', 'f', 'False', 'false']:
        from ui.cli import CLI
        view = CLI(model)

    else:
        from ui.gui import GUI
        view = GUI(model)

    view.run()

if __name__ == '__main__':
    main()
