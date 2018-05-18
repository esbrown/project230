import sys
import csv

def cleanData(data):
	with open(data, 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter='*', quotechar='|')
		with open('regTweets.csv', 'w') as writecsv:
			fieldnames = ['word_vec']
			writer = csv.DictWriter(writecsv, fieldnames = fieldnames)
			writer.writeheader()
			for row in reader:
				if len(row) == 1:
					cleanedText = row[0].replace("\"", "") ###get rid of quotation marks
					wordVec = row[0].split(' ')
					print wordVec
					writer.writerow({'word_vec': wordVec})

if __name__ == "__main__":
	data = sys.argv[1]
	cleanData(data)