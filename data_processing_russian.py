import sys
import csv
import nltk

def cleanData(data):
	maxWords = 50
	with open(data, 'rU') as csvfile:
		reader = csv.reader(csvfile)
		with open('russianTweets.csv', 'w') as writecsv:
			fieldnames = ['word_vec']
			writer = csv.DictWriter(writecsv, fieldnames = fieldnames)
			writer.writeheader()
			for row in reader:
				if len(row) == 1:
					cleanedText = row[0].replace("\"", "") ###get rid of quotation marks and TODO: others?
					cleanedText = cleanedText.replace("\'", "")
					cleanedText = cleanedText.replace("*", " ")
					wordVec = ntlk.word_tokenize(cleanedText)
					unkTokens = maxWords - len(wordVec)
					if unkTokens < 0:
						print 'Increase MaxWords'
					wordVec = wordVec + ['<UNK>' for i in range(unkTokens)]
					writer.writerow({'word_vec': wordVec})

if __name__ == "__main__":
	data = sys.argv[1]
	cleanData(data)