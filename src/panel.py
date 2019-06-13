from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from src.runner import *


class Root(Tk):
    lyrics = []

    def __init__(self):
        super(Root, self).__init__()
        self.title("Music sentiment analysis")
        self.minsize(700, 700)
        self.wm_iconbitmap('resources/icon.ico')

        # add buttons for ML algorithms

        # classify one song
        self.songTitle = ttk.LabelFrame(self, text="Classify one song")
        self.songTitle.grid(column=1, row=1)
        self.classifysong()

        # classify album
        self.albumTitle = ttk.LabelFrame(self, text="Classify album")
        self.albumTitle.grid(column=1, row=5)
        self.classifyalbum()

        # run algorithms
        self.classificationTitle = ttk.LabelFrame(self, text="Run classification algorithms")
        self.classificationTitle.grid(column=1, row=10)
        self.classification()


    def classifysong(self):
        self.songLabel = ttk.Label(self.songTitle, text="Insert song lyrics")
        self.songLabel.grid(column=1, row=2)
        self.songText = Text(self.songTitle)
        self.songText.grid(column=2, row=2)
        self.predictButton = ttk.Button(self.songTitle, text="Predict sentiment", command=self.predictsong)
        self.predictButton.grid(column=2, row=3)


    def predictsong(self):
        textInput = self.songText.get("1.0", "end-1c")
        result = predictMNBrunnerText(textInput)[0]
        self.resultLabel = ttk.Label(self.songTitle, text="")
        self.resultLabel.grid(column=2, row=4)
        self.resultLabel.configure(text="Sentiment: " + result)


    def classifyalbum(self):
        self.labelAlbumName = ttk.Label(self.albumTitle, text="Album name")
        self.labelAlbumName.grid(column=1, row=6)
        self.inputAlbumName = ttk.Entry(self.albumTitle)
        self.inputAlbumName.grid(column=2, row=6)
        self.labelSingerName = ttk.Label(self.albumTitle, text="Singer name")
        self.labelSingerName.grid(column=1, row=7)
        self.inputSingerName = ttk.Entry(self.albumTitle)
        self.inputSingerName.grid(column=2, row=7)
        self.openLabel = ttk.Label(self.albumTitle, text="Open album lyrics")
        self.openLabel.grid(column=1, row=8)
        self.fileButton = ttk.Button(self.albumTitle, text="Classify album", command=self.openfile)
        self.fileButton.grid(column=2, row=8)


    def openfile(self):
        self.fileNames = filedialog.askopenfilenames(initialdir="/Desktop", title="Select .txt files",
                                               filetypes=(("txt", "*.txt"), ("All files", "*.*")))
        for fileName in self.fileNames:
            lyric = open(fileName, 'r').read()
            self.lyrics.append(lyric)

        results = predictMNBrunnerVector(self.lyrics)
        noPositive = 0
        noNegative = 0
        for result in results:
            if result == 'positive':
                noPositive += 1
            else:
                noNegative += 1
        self.albumResult = ttk.Label(self.albumTitle, text="")
        self.albumResult.grid(column=2, row=9)
        if noPositive >= noNegative:
            self.albumResult.configure(text="Sentiment: positive")
        else:
            self.albumResult.configure(text="Sentiment: negative")


    def classification(self):
        self.mnbbutton = ttk.Button(self.classificationTitle, text="Run MNB", command=self.runMNBGui)
        self.mnbbutton.grid(column=1, row=11)
        self.svmbutton = ttk.Button(self.classificationTitle, text="Run SVM", command=runSVM)
        self.svmbutton.grid(column=2, row=11)
        self.dtcbutton = ttk.Button(self.classificationTitle, text="Run DTC", command=runDTC)
        self.dtcbutton.grid(column=3, row=11)
        self.algorithmResult = ttk.Label(self.classificationTitle, text="")
        self.algorithmResult.grid(column=2, row=12)


    def runMNBGui(self):
        result = runMNB()
        self.algorithmResult.configure(text=result)


if __name__ == '__main__':
    root = Root()
    root.mainloop()
