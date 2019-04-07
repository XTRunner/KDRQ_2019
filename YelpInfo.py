import csv

count = 0
numCount = 0
latList = []
lonList = []
with open('yelp.csv') as myfile:
	reader = csv.reader(myfile)
	for row in reader:
		lat = row[0]
		lon = row[1]
		latList.append(lat)
		lonList.append(lon)
		newKeywords = []
		keyList = str(row[2]).split(',')
		count += len(keyList)
		numCount += 1

print numCount
print count/float(numCount)

print max(latList), min(latList), max(lonList), min(lonList)

