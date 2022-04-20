from model.model import Decompiler


class CLI:
    def __init__(self, model: Decompiler):
        self.model = model
        
    def run(self):
        print('path to binary file: ', end='')
        filename = input()
        asm = self.model.open_binary_file(filename)
        self.model.decompile()
        output = f"""
ORIGINAL:
{asm}

SEGMENTAION:
{self.model.get_segmented_asm()}

MASKED C:
{self.model.get_masked_c()}

RECONSTRUCTED C:
{self.model.get_reconstructed_c()}
        """
        print(output)
