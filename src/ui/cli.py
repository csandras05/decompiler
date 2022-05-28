from model.model import Decompiler


class CLI:
    def __init__(self, model: Decompiler):
        self.model = model
        
    def run(self):
        filename = input('path to binary file: ')
        match filename:
            case 'exit':
                print("Closing program...")
            case _:
                try:
                    asm = self.model.open_binary_file(filename)
                    print("Decompiling code...\nIt may take a few seconds.")
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
                except AttributeError:
                    print(f"File '{filename}' not found!")
                    print("Closing program...")
