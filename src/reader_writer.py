import csv
import codecs
import pandas as pd

def read_csv(path):
    # data_list = list()
    # with open(path) as csvFile:
    #     reader = csv.DictReader(csvFile)
    #     for row in reader:
    #         data = {'ID': row['ID'], 'Title': row['Title'], 'Singer': row['Singer'],
    #                 'Words': row['Words'], 'Sentiment': row['Sentiment'], 'Lyrics': row['Lyrics']}
    #         data_list.append(data)

    data_list = pd.read_csv(path)
    # print("Shape of the dataset:")
    # print(data_list.shape)
    # print("Column names:")
    # print(data_list.columns)
    # print("Datatype of each column:")
    # print(data_list.dtypes)
    # print("Few dataset entries:")
    # print(data_list.head())
    # DATASET SUMMARY
    data_list.describe(include='all')
    return data_list

# def write_arff(path, data):
#     file = codecs.open(path, "w", "utf-8")
#     file.write("@relation sentiment-analysis-music\n")
#     file.write("@attribute ID numeric\n")
#     file.write("@attribute Title string\n")
#     file.write("@attribute Words numeric\n")
#     file.write("@attribute Lyrics string\n")
#     file.write("@attribute Sentiment {positive, negative}\n")
#     file.write("@data\n")
#     for d in data:
#         text = d['ID'] + ",\"" + d['Title'] + "\"," + d['Words'] + ",\"" + d['Lyrics'] + "\"," + d['Sentiment']
#         file.write(text + "\n")
#
#     file.close()