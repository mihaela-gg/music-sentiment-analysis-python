from tkinter import *
from src.runner import *

root = Tk()


def predictSong():
    textInput = songText.get("1.0", "end-1c")
    print(textInput)
    result = predictMNBrunnerText(textInput)[0]
    print(result)
    predictionLabel.config(text="Result: " + result)

topFrame = Frame(root)
topFrame.pack()
bottomFrame = Frame(root)
bottomFrame.pack(side=BOTTOM)

MNBbutton = Button(topFrame, text="Run MNB", command=runMNB)
SVMbutton = Button(topFrame, text="Run SVM", command=runSVM)
DTCbutton = Button(topFrame, text="Run DTC", command=runDTC)
MNBbutton.pack(side=LEFT)
SVMbutton.pack(side=LEFT)
DTCbutton.pack(side=LEFT)
mlResultLabel = Label(topFrame)
mlResultLabel.pack()

songLabel = Label(bottomFrame, text="Enter the song")
songText = Text(bottomFrame)
predictButton = Button(bottomFrame, text="Predict sentiment", command=predictSong)
songLabel.grid(row=0, column=0)
songText.grid(row=0, column=1)
predictButton.grid(row=1, column=1)
predictionLabel = Label(bottomFrame)
predictionLabel.grid(row=2, column=1)

root.mainloop()
