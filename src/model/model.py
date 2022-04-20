import os
import random
import re
import subprocess
from typing import Any, List

import utils
from flax_models.segmentation import Segmentation, SegmentationModel
from flax_models.translation import Seq2seq, Translation
from reconstruct import reconstruct


class Decompiler:
    
    def __init__(self):
        self.asm = None
        self.asm_embeddings = None
        self.segmentation_indices = None
        self.masked_c = None
        self.reconstructed_c = None
        
        self.palmtree = utils.load_palmtree()
        self.segmentation = Segmentation(f'{os.path.dirname(__file__)}/flax_models/segmentation.params',
                                         SegmentationModel(hidden_size=256))
        self.translation = Translation(f'{os.path.dirname(__file__)}/flax_models/translation.params',
                                       Seq2seq(256, 18, 1, 29)) # TODO: place values in a conf file
    
    def decompile(self):
        self.segmentation_indices = self.segmentation.get_segmentation(self.asm_embeddings)
        
        embeddings_blocks = [self.asm_embeddings[cur:nxt] for (cur, nxt) in zip(self.segmentation_indices,
                                                                                self.segmentation_indices[1:])]
        
        self.masked_c = self.translation.translate(embeddings_blocks[1:-1])
        
        asm_blocks = ['\n'.join(self.asm[cur:nxt]) for (cur, nxt) in zip(self.segmentation_indices,
                                                                         self.segmentation_indices[1:])]
        
        self.reconstructed_c = reconstruct.retrieve(self.masked_c, asm_blocks[1:-1])
        
        
    def get_segmented_asm(self) -> str:
        return '\n\n'.join('\n'.join(self.asm[cur:nxt]) for (cur, nxt) in zip(self.segmentation_indices,
                                                                              self.segmentation_indices[1:]))
    
    def get_masked_c(self) -> str:
        return '\n'.join(self.masked_c)
    
    def get_reconstructed_c(self) -> str:
        return self.reconstructed_c
    
    def open_binary_file(self, filename: str) -> str:
        """Open and extract the X86 assembly from the binary of a gcc compiled C code

            Disassembling the binary file using objdump and then extracting the assembly
            instructions corresponding to the main function using regexp

        Args:
            filename (str): gcc compiled C code (e.g. a.out)

        Returns:
            str: X86 assembly instructions of the main function
        """
        objdumped = utils.objdump(binary=filename,
                                  flags='-S -l --no-show-raw-insn -M intel',
                                  output=subprocess.PIPE).stdout.decode()
        main = utils.extract_fun(objdumped=objdumped, fun_name='main')
        self.asm = list(utils.extract_asm(main))
        self.asm_embeddings = utils.encode(self.palmtree, self.asm)
        return '\n'.join(self.asm)
