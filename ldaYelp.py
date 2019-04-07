import csv
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import re
from stop_words import get_stop_words
from collections import Counter
import math
from random import randint

def stem_words(text):
    """
    Function to stem words, so plural and singular are treated the same
    """
    stemmer = PorterStemmer()
    try:
        text = [stemmer.stem(word.lower()) for word in text]
        text = [str(word.lower()) for word in text if len(word) > 1] # make sure we have no 1 letter words
    except:
    	pass
    return text

def recreateDict():
	texts = []
	freq = []
	en_stop = get_stop_words('en')
	#gradient.csv
	with open('gradient.csv') as myfile:
		csv_reader = csv.reader(myfile)
		for row in csv_reader:
			x = ",".join(row)
			dishList = x.split(',')
			dishList = stem_words(dishList)
			newList = []
			for dish in dishList:
				dish = re.sub('[^A-Za-z]', '', str(dish))
				#print dish
			#tokens = tokenizer.tokenize(raw)
			# remove stop words from tokens
			#stopped_tokens = [i for i in raw if not i in en_stop]
			# stem tokens
			#stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
			# add tokens to list
				if not dish in en_stop:
					dish = dish.lower()
					newList.append(dish)
					freq.append(dish)
			#print newList			
			texts.append(newList)
	
	#print "Filter High Freq Words"
	top = Counter(freq).most_common(1000)
	#print top
	topList = []
	for i in top:
		topList.append(i[0])
	topList.remove('allrecip')
	topList.remove('recip')
	topList.remove('martha')
	#topList.remove('re')
	topList.remove('stewart')
	topList.remove('myrecip')
	topList.remove('recipe')
	
	final = []
	for i in texts:
		partFinal = []
		for j in i:
			if j in topList:
				partFinal.append(j)
		final.append(partFinal)
	
	# turn our tokenized documents into a id <-> term dictionary
	dictionary = corpora.Dictionary(final)
	#corpus = [dictionary.doc2bow(text) for text in final]
	return dictionary, topList

def main():
	ldamodel = models.ldamodel.LdaModel.load('lda.model')
	#print ldamodel.print_topics()
	dictionary, topList = recreateDict()
	with open('yelp.csv') as myfile:
		reader = csv.reader(myfile)
		for row in reader:
			lat = row[0]
			lon = row[1]
			newKeywords = []
			keyList = str(row[2]).split(',')
			for key in keyList:
				key = re.sub('[^A-Za-z]', '', str(key))
				oldKey = key.split()
				for i in oldKey:
					newKeywords.append(i)
			newKeyList = stem_words(newKeywords)
			if 'restaur' in newKeyList:
				newKeyList.remove('restaur')
			for j in newKeyList:
				if j not in topList:
					newKeyList.remove(j)
			if newKeyList:
				try:
					bow_vector = dictionary.doc2bow(newKeyList)
					#print bow_vector
					#print ldamodel.get_document_topics(bow_vetor)
					#topics = model.transform(bow_vector)
					#ldamodel.get_term_topics('taco')
					LDAres = ldamodel[bow_vector]
					ldaVector = []
					for i in range(10):
						ldaVector.append(float(format(LDAres[i][1], '.1f')))
					'''
					format(LDAres[i][1], '.1f')
					for index, score in sorted(ldamodel[bow_vector], key=lambda tup: -1*tup[1]):
						print("Score: {}\t Topic: {}".format(score, ldamodel.print_topic(index, 8)))
					print "//////////////////////////////////////"
					'''
					#print ldaVector
					while (sum(ldaVector) < 1):
						ldaVector[randint(0, 9)] += 0.1
					while (sum(ldaVector) > 1):
						ldaVector[ldaVector.index(max(ldaVector))] -= 0.1
					
					#ldaVector = [float(x)/10 for x in ldaVector]
					
					with open('PoIlda.csv', 'a') as wfile:
						spamwriter = csv.writer(wfile)
						spamwriter.writerow([lat, lon, ldaVector])
				except Exception as e: 
					print(e)
		
if __name__ == '__main__':
	main()
				