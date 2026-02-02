from PyQt5.QtWidgets import *
import sys
import subprocess
import shutil
import numpy as np
try:
    import simpleaudio as sa
    _HAS_SIMPLEAUDIO = True
except Exception:
    _HAS_SIMPLEAUDIO = False

class BeepSound(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('삑 소리 내기')
        self.setGeometry(100, 100, 500, 100)

        shortBeepButton = QPushButton('짧은 삑 소리', self)
        longBeepButton = QPushButton('긴 삑 소리', self)
        quitButton = QPushButton('나가기', self)
        self.label = QLabel('환영합니다.', self)

        shortBeepButton.setGeometry(10, 10, 100, 30)
        longBeepButton.setGeometry(110, 10, 100, 30)
        quitButton.setGeometry(210, 10, 100, 30)
        self.label.setGeometry(10, 50, 300, 30)

        shortBeepButton.clicked.connect(self.shortBeepfunction)
        longBeepButton.clicked.connect(self.longBeepfunction)
        quitButton.clicked.connect(QApplication.instance().quit)

    def shortBeepfunction(self):
        self.label.setText('주파수 1000Hz, 0.5초 동안 삑 소리')
        self.play_beep(short=True)

    def longBeepfunction(self):
        self.label.setText('주파수 1000Hz, 3초 동안 삑 소리')
        self.play_beep(long=True)

    def play_beep(self, short=False, long=False):
        """Cross-platform: Windows uses winsound, else synth+simpleaudio or fallback players."""
        if sys.platform.startswith('win'):
            import winsound
            duration = 500 if short else 3000 if long else 500
            winsound.Beep(1000, duration)
            return

        freq = 1000
        duration_ms = 500 if short else 3000 if long else 500

        if _HAS_SIMPLEAUDIO:
            fs = 44100
            t = np.linspace(0, duration_ms / 1000.0, int(fs * duration_ms / 1000.0), False)
            tone = np.sin(2 * np.pi * freq * t)
            audio = (tone * 32767).astype(np.int16)
            sa.play_buffer(audio, 1, 2, fs)
            return

        if sys.platform == 'darwin':
            sound = '/System/Library/Sounds/Glass.aiff' if short else '/System/Library/Sounds/Sosumi.aiff'
            if shutil.which('afplay'):
                subprocess.Popen(['afplay', sound])
            else:
                subprocess.run(['osascript', '-e', 'beep'])
            return

        for cmd in (['paplay'], ['aplay'], ['play']):
            if shutil.which(cmd[0]):
                subprocess.Popen(cmd + ['beep.wav'])
                return

        print('\a')

app = QApplication(sys.argv)
win = BeepSound()
win.show()
app.exec_()