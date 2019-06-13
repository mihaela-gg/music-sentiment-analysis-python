from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.runner import *


class Root(Tk):
    lyrics = []

    def __init__(self):
        super(Root, self).__init__()
        self.title("Music sentiment analysis")
        self.minsize(700, 700)
        self.wm_iconbitmap("resources/icon.ico")

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
        self.labelAlbumName = ttk.Label(self.albumTitle, text="Album")
        self.labelAlbumName.grid(column=1, row=6)
        self.inputAlbumName = ttk.Entry(self.albumTitle)
        self.inputAlbumName.grid(column=2, row=6)
        self.labelSingerName = ttk.Label(self.albumTitle, text="Singer/Band")
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
            lyric = open(fileName, "r").read()
            self.lyrics.append(lyric)

        results = predictMNBrunnerVector(self.lyrics)
        noPositive = 0
        noNegative = 0
        for result in results:
            if result == "positive":
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
        self.svmbutton = ttk.Button(self.classificationTitle, text="Run SVM", command=self.runSVMGui)
        self.svmbutton.grid(column=2, row=11)
        self.dtcbutton = ttk.Button(self.classificationTitle, text="Run DTC", command=self.runDTCGui)
        self.dtcbutton.grid(column=3, row=11)


    def runMNBGui(self):
        prediction = predictMNB(x_train, y_train, x_test)
        self.showPredictions(prediction)


    def runSVMGui(self):
        prediction = predictSVM(x_train, y_train, x_test)
        self.showPredictions(prediction)


    def runDTCGui(self):
        prediction = predictDTC(x_train, y_train, x_test)
        self.showPredictions(prediction)


    def showPredictions(self, prediction):
        confusionMatrix = confusion_matrix(y_test, prediction)
        accuracy = round(accuracy_score(y_test, prediction) * 100, 2)
        precisionPos = round(precision_score(y_test, prediction, average="binary", pos_label="positive"), 2)
        precisionNeg = round(precision_score(y_test, prediction, average="binary", pos_label="negative"), 2)
        recallPos = round(recall_score(y_test, prediction, average="binary", pos_label="positive"), 2)
        recallNeg = round(recall_score(y_test, prediction, average="binary", pos_label="negative"), 2)
        f1Pos = round(f1_score(y_test, prediction, average="binary", pos_label="positive"), 2)
        f1Neg = round(f1_score(y_test, prediction, average="binary", pos_label="negative"),2)

        # show confusion matrix
        self.labelConfusionMatrix = ttk.Label(self.classificationTitle, text="Confusion matrix:")
        self.labelConfusionMatrix.grid(column=1, row=12)
        self.treeConfusionMatrix = ttk.Treeview(self.classificationTitle, height=2)
        self.treeConfusionMatrix["columns"] = ("negative", "positive")
        self.treeConfusionMatrix.column("#0", width=50, minwidth=50)
        self.treeConfusionMatrix.column("negative", width=50, minwidth=50)
        self.treeConfusionMatrix.column("positive", width=50, minwidth=50)
        self.treeConfusionMatrix.heading("negative", text="-")
        self.treeConfusionMatrix.heading("positive", text="+")
        self.treeConfusionMatrix.insert("", 1, text="-", values=(confusionMatrix[1][0], confusionMatrix[1][1]))
        self.treeConfusionMatrix.insert("", 2, text="+", values=(confusionMatrix[0][0], confusionMatrix[0][1]))
        self.treeConfusionMatrix.grid(column=2, row=12)

        # show accuracy
        self.labelAccuracy = ttk.Label(self.classificationTitle, text="Accuracy: " + accuracy.__str__())
        self.labelAccuracy.grid(column=2, row=13)

        # show classification report
        self.labelClassificatonReport = ttk.Label(self.classificationTitle, text="Classification report")
        self.labelClassificatonReport.grid(column=1, row=14)
        self.treeClassificationReport = ttk.Treeview(self.classificationTitle, height=2)
        self.treeClassificationReport["columns"] = ("precision", "recall", "f1score")
        self.treeClassificationReport.column("#0", width=70, minwidth=70)
        self.treeClassificationReport.column("precision", width=70, minwidth=70)
        self.treeClassificationReport.column("recall", width=70, minwidth=70)
        self.treeClassificationReport.column("f1score", width=70, minwidth=70)
        self.treeClassificationReport.heading("precision", text="Precision")
        self.treeClassificationReport.heading("recall", text="Recall")
        self.treeClassificationReport.heading("f1score", text="F1-score")
        self.treeClassificationReport.insert("", 1, text="-", values=(precisionNeg, recallNeg, f1Neg))
        self.treeClassificationReport.insert("", 2, text="+", values=(precisionPos, recallPos, f1Pos))
        self.treeClassificationReport.grid(column=2, row=14)


if __name__ == "__main__":
    root = Root()
    root.mainloop()
