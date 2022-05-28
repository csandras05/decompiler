import PySimpleGUI as sg  # type: ignore
from model.model import Decompiler


class GUI:
    def __init__(self, model: Decompiler):
        sg.theme('Topanga')
        self.model = model
        layout = [[sg.Multiline(key='-INPUT-', disabled=True, size=(50, 30), expand_x=True, expand_y=True),
                   sg.Multiline(key='-OUTPUT-', disabled=True, size=(50, 30), expand_x=True, expand_y=True)],
        
                  [sg.Input(key='-FILE-', expand_x=True),
                   sg.FileBrowse(),
                   sg.Button('Submit')],
        
                  [sg.Button('Decompile', disabled=True, key='-DECOMPILE-'),
                   sg.Button('Segmentation', disabled=True, key='-SEGMENTATION-'),
                   sg.Button('Masked C', disabled=True, key='-MASKED_C-'),
                   sg.Button('Reconstructed C', disabled=True, key='-RECONSTRUCTED_C-')]]
        
        self.window = sg.Window('Decompilaton', layout, resizable=True, finalize=True, font=("Helvetica 16"),
                                auto_size_buttons=True)
        self.window['-FILE-'].expand(True)
        
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
                    self.window['-SEGMENTATION-'].update(button_color=('#E7C855', '#284B5A'))
                    self.window['-MASKED_C-'].update(button_color=('#E7C855', '#284B5A'))
                    self.window['-RECONSTRUCTED_C-'].update(button_color=('#E7C855', '#284B5A'))
                    
                    filename = values['-FILE-']
                    
                    try:
                        text = self.model.open_binary_file(filename)
                        self._enable('-DECOMPILE-')
                    except Exception as e:
                        text = f'{filename}\nNOT FOUND OR FILE FORMAT NOT SUPPORTED!'
                        self._disable('-DECOMPILE-')
                                                
                    self.window['-INPUT-'].update(text)
                    self._disable('-SEGMENTATION-', '-MASKED_C-', '-RECONSTRUCTED_C-')
                    self.window['-OUTPUT-'].update('')
            
                case '-DECOMPILE-':
                    self.window['-SEGMENTATION-'].update(button_color=('#E7C855', '#284B5A'))
                    self.window['-MASKED_C-'].update(button_color=('#E7C855', '#284B5A'))
                    self.window['-RECONSTRUCTED_C-'].update(button_color=('#E7C855', '#284B5A'))
                    self.window['-OUTPUT-'].update('Decompiling code...\nIt may take a few seconds.')
                    self.window.refresh()
                    self.model.decompile()
                    self.window['-OUTPUT-'].update('Decompilation done.\nClick on the buttons below to see the result.')
                    self._enable('-SEGMENTATION-', '-MASKED_C-', '-RECONSTRUCTED_C-')
                
                case '-SEGMENTATION-':
                    self.window['-SEGMENTATION-'].update(button_color=('#284B5A', '#E7C855'))
                    self.window['-MASKED_C-'].update(button_color=('#E7C855', '#284B5A'))
                    self.window['-RECONSTRUCTED_C-'].update(button_color=('#E7C855', '#284B5A'))
                    self.window['-OUTPUT-'].update(self.model.get_segmented_asm())
                  
                case '-MASKED_C-':
                    self.window['-SEGMENTATION-'].update(button_color=('#E7C855', '#284B5A'))
                    self.window['-MASKED_C-'].update(button_color=('#284B5A', '#E7C855'))
                    self.window['-RECONSTRUCTED_C-'].update(button_color=('#E7C855', '#284B5A'))
                    self.window['-OUTPUT-'].update(self.model.get_masked_c())
                    
                case '-RECONSTRUCTED_C-':
                    self.window['-SEGMENTATION-'].update(button_color=('#E7C855', '#284B5A'))
                    self.window['-MASKED_C-'].update(button_color=('#E7C855', '#284B5A'))
                    self.window['-RECONSTRUCTED_C-'].update(button_color=('#284B5A', '#E7C855'))
                    self.window['-OUTPUT-'].update(self.model.get_reconstructed_c())
            
        self.window.close()
