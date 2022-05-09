import json
import os
import subprocess

from model import utils
from model.flax_models.segmentation import Segmentation, SegmentationModel
from model.flax_models.translation import Seq2seq, Translation
from model.reconstruct import reconstruct

CUR_DIR = os.path.dirname(__file__)

class Decompiler:
    
    def __init__(self, config):
        self.asm = None
        self.asm_embeddings = None
        self.segmentation_indices = None
        self.masked_c = None
        self.reconstructed_c = None
        
        self.palmtree = utils.load_palmtree()
        
        segmentation_model = SegmentationModel(hidden_size=config['segmentation']['hidden_size'])
        self.segmentation = Segmentation(params_file=f"{CUR_DIR}/{config['segmentation']['model_path']}",
                                         model=segmentation_model,
                                         max_len=config['segmentation']['max_len'],
                                         embedding_size=config['embedding_size'])
        
        with open(f"{CUR_DIR}/{config['translation']['vocab_path']}", 'r') as f:
            vocab = json.load(f)
        
        translation_model = Seq2seq(hidden_size=config['translation']['hidden_size'],
                                    vocab_size=len(vocab),
                                    sos_id=vocab['<SOS>'],
                                    max_output_len=config['translation']['max_output_len'])
        self.translation = Translation(params_file=f"{CUR_DIR}/{config['translation']['model_path']}",
                                       model=translation_model,
                                       vocab=vocab,
                                       max_input_len=config['translation']['max_input_len'],
                                       embedding_size=config['embedding_size'])
                                       
    
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
        objdumped = utils.objdump(binary=filename,
                                  flags='-S -l --no-show-raw-insn -M intel',
                                  output=subprocess.PIPE).stdout.decode()
        main = utils.extract_fun(objdumped=objdumped, fun_name='main')
        self.asm = list(utils.extract_asm(main))
        self.asm_embeddings = utils.encode(self.palmtree, self.asm)
        return '\n'.join(self.asm)
