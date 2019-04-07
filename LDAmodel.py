import csv
import sys
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import random
import math
import re
from collections import Counter

def stem_words(text):
    """
    Function to stem words, so plural and singular are treated the same
    """
    stemmer = PorterStemmer()
    try:
        text = [stemmer.stem(word.lower()) for word in text]
        text = [word.lower() for word in text if len(word) > 1] # make sure we have no 1 letter words
    except:
    	pass
    return text

def sameMenu():
	with open('MenuPage.csv') as myfile:
		csv_reader = csv.reader(myfile, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				line_count += 1
			elif line_count == 1:
				page_menu = {row[1]: [row[0]]}
				line_count += 1
			else:
				flag = 0
				for i in page_menu.keys():
					if row[1] == i:
						flag = 1
						page_menu[i].append(row[0])
						break
				if (flag == 0):
					page_menu[row[1]] = [row[0]]
				line_count += 1
			print "Building Menu Page:" + str(line_count) + "/66938"
    	return page_menu
    		
def samePage(page_menu):
	with open('Dish.csv') as dish:
		dish_reader = csv.reader(dish, delimiter=',')
		line_count = 0
		for row in dish_reader:
			if line_count == 0:
				line_count += 1
			elif line_count == 1:
				dishName = {row[0]:row[1]}
				line_count += 1
			else:
				if row[1] and row[0]:
					dishName[row[0]] = row[1]
	
	dishPage = {}
	for i in page_menu.keys():
		dishPage[i] = []
	
	with open('MenuItem.csv') as menu:
		menu_reader = csv.reader(menu, delimiter=',')
		line_count = 0
		for row in menu_reader:
			if line_count == 0:
				line_count += 1
			elif line_count == 1:
				name = dishName[row[4]]
				for menuID, pageID in page_menu.items():
					if row[1] in pageID:
						dishPage[menuID].append(name)
						line_count += 1
						break
			else:
				flag = 0
				try:
					name = dishName[row[4]]
				except:
					line_count += 1
					continue
				for menuID, pageID in page_menu.items():
					if row[1] in pageID:
						dishPage[menuID].append(name)
						break
				line_count += 1
			if line_count > 500000:
				break
			print str(line_count) + "/500000"
	return dishPage
	
def LDA(trainSet, testSet, topics=10, times=200):
	# topics: # of topics in the result
	# times: # of passes during training
	#tokenizer = RegexpTokenizer(r'\w+')

	# create English stop words list
	en_stop = get_stop_words('en')

	# Create p_stemmer of class PorterStemmer
	#p_stemmer = PorterStemmer()
	 
	# create sample documents
	#r1 = "1, 2, 3"
	#r2 = "2, 3, 4"
	#r3 = "1, 3, 5"
	#r4 = "2, 4, 5"
	#r5 = "1, 5, 6"
 
	# compile sample documents into a list
	#r_set = [r1, r2, r3, r4, r5]
	
	#print r_set

	# list for tokenized documents in loop
	texts = []
	freq = []
	
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
	
	print "Filter High Freq Words"
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
	topList.remove('recipes')
	topList.remove('street')
	topList.remove('epicuri')
	topList.remove('edamam')
	
	final = []
	for i in texts:
		partFinal = []
		for j in i:
			if j in topList:
				partFinal.append(j)
		final.append(partFinal)
		
	# turn our tokenized documents into a id <-> term dictionary
	dictionary = corpora.Dictionary(final)
	
	#print dictionary
	# convert tokenized documents into a document-term matrix
	corpus = [dictionary.doc2bow(text) for text in final]
	#print corpus
	print "Train Model"
	# generate LDA model
	flag = False
	while not flag:
		ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topics, id2word = dictionary, passes=times)
		ldamodel.save('lda.model')
		flag = True
		japanFlag = False
		for entry in ldamodel.print_topics():
			print entry
			dishes = str(entry[1]).split("+")
			weight = []
			name = []
			for dish in dishes:
				dish = str(dish).split("*")
				#print dish[0]
				weight.append(float(dish[0]))
				name.append(re.sub('[^A-Za-z]', '', str(dish[1])))
			if max(weight) <= 0.002:
				flag = False
				break
			if "thai" in name and "chines" in name:
				flag = False
				break
			if "thai" in name and "indian" in name:
				flag = False
				break
			if "japanes" in name:
				japanFlag = True
		if not japanFlag:
			flag = False	
	#print ldamodel.print_topics()
	'''
	#unseen_document = ['k1','k2','k3']
	testSet = stem_words(testSet)
	#print testSet
	bow_vector = corpora.Dictionary(final).doc2bow(testSet)

	for index, score in sorted(ldamodel[bow_vector], key=lambda tup: -1*tup[1]):
		print("Score: {}\t Topic: {}".format(score, ldamodel.print_topic(index, 8)))
	'''


def main():
	"""
	menuDict = sameMenu()
	dishDict = samePage(menuDict)
	with open('mergeRes.csv', 'wb') as csvfile:
		for i in dishDict.values():
			if i:
				spamwriter = csv.writer(csvfile, delimiter=',')
				spamwriter.writerow(i)
	"""
	#LDA('mergeRes.csv', ['piazza', 'spaghetti', 'sauage'])
	LDA('mergeRes.csv', ['burger', 'Fried', 'Chicken', 'cheese'])
	
if __name__ == "__main__":
	main()