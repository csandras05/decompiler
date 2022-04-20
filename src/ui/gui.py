import PySimpleGUI as sg  # type: ignore
from model.model import Decompiler


class GUI:
    def __init__(self, model: Decompiler):
        self.model = model
        layout = [[sg.Multiline(key='-INPUT-', disabled=True, size=(40, 20)),
                        sg.Multiline(key='-OUTPUT-', disabled=True, size=(40, 20))],
          
                       [sg.Input(key='-FILE-'),
                        sg.FileBrowse(),
                        sg.Button('Submit')],
          
                       [sg.Button('Decompile', disabled=True),
                        sg.Button('Segmentation', disabled=True),
                        sg.Button('Masked C', disabled=True),
                        sg.Button('Reconstructed C', disabled=True)]]
        
        self.window = sg.Window('Decompilaton', layout)

        
    def _disable(self, *widgets):
        for w in widgets:
            self.window[w].update(disabled=True)
            
    def _enable(self, *widgets):
        for w in widgets:
            self.window[w].update(disabled=False)
        
    def run(self):
        while True:
            event, values = self.window.read()
            
            match event:
                case sg.WINDOW_CLOSED:
                    break
                
                case 'Submit':
                    filename = values['-FILE-']
                    
                    try:
                        text = self.model.open_binary_file(filename)
                        self._enable('Decompile')
                    except Exception as e:
                        text = f'{filename}\nNOT FOUND OR FILE FORMAT NOT SUPPORTED!'
                        self._disable('Decompile')
                                                
                    self.window['-INPUT-'].update(text)
                    self._disable('Segmentation', 'Masked C', 'Reconstructed C')
                    self.window['-OUTPUT-'].update('')
            
                case 'Decompile':
                    self.model.decompile()
                    self._enable('Segmentation', 'Masked C', 'Reconstructed C')
                
                case 'Segmentation':
                    self.window['-OUTPUT-'].update(self.model.get_segmented_asm())
                  
                case 'Masked C':
                    self.window['-OUTPUT-'].update(self.model.get_masked_c())
                    
                case 'Reconstructed C':
                    self.window['-OUTPUT-'].update(self.model.get_reconstructed_c())
            
        self.window.close()
