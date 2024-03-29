#import osmnx as ox
#import pandas as pd
#import geopandas as gpd
import csv
import os
import random
import sys
import timeit
#from descartes import PolygonPatch
#from shapely.geometry import Point, c, MultiPolygon


# sptSet: shortest path tree
# dist: distance list
class Graph():
	
	# Graph as two-dimension list
	def __init__(self, vertices):
		self.V = vertices
		self.graph = [[0 for column in range(vertices)] for row in range(vertices)]

	# Find the next node with minimum distance and not included into sptSet
	# If find, return index. If not, then rest of nodes are unconnnected
	def minDistance(self, dist, sptSet):

		# Initilaize minimum distance for next node
		min = 2500
		flag = False
		# Search not nearest vertex not in the 
		# shortest path tree
		for v in range(self.V):
			if dist[v] < min and sptSet[v] == False:
				flag = True
				min = dist[v]
				min_index = v
		
		if flag == True:
			return flag, min_index
		else:
			return flag, 'unconnected'

	def dijkstra(self, src):
		# dist: [0, infinity, infinity,...]
		dist = [sys.maxint] * self.V
		dist[src] = 0
		edgeCount = 0
		
		# sptSet: [F,F,F,...]
		sptSet = [False] * self.V

		for cout in range(self.V):
			flag, u = self.minDistance(dist, sptSet)
			# Put the minimum distance vertex in the 
			# shotest path tree
			if flag == True:
				# Include that node into sptSet
				sptSet[u] = True
				
				# node has not included in sptSet and current value is greater than new distance
				for v in range(self.V):
					if self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]:
						edgeCount = edgeCount + 1
						dist[v] = dist[u] + self.graph[u][v]
			else:
				# Unconnected. Connected part already done
				break
		#print dist
		return dist, edgeCount

def getMap(city, type):
	#gdf = ox.gdf_from_place(city)
	# get the street network within this bounding box
	#west, south, east, north = gdf.unary_union.buffer(0.1).bounds
	G = ox.graph_from_place(city, network_type=type, simplify=True)
	
	network = []
	with open(city+type+'edge.csv', 'wb') as csvfile:
		for start, end, nothing, data in G.edges(keys=True, data=True):
			spamwriter = csv.writer(csvfile, delimiter='|')
			spamwriter.writerow([str(start),str(end),float(data['length'])])
			network.append([str(start),str(end),float(data['length'])])
	
	nodeSet = set()	
	with open(city+type+'node.csv', 'wb') as csvfile:
		for node, data in G.nodes(data=True):
			spamwriter = csv.writer(csvfile, delimiter='|')
			spamwriter.writerow([str(node),float(data['x']),float(data['y'])])
			nodeSet.add(str(node))
	#print network
	return network, nodeSet
	
def neighbor(network ,nodeSet):
	neighborList = {}
	for node in nodeSet:
		nodeNeighbor = {}
		for pair in network:
			if node == pair[0]:
				nodeNeighbor[pair[1]] = pair[2]
			elif node == pair[1]:
				nodeNeighbor[pair[0]] = pair[2]
			else:
				continue
		neighborList[node] = nodeNeighbor
	#print neighborList
	return neighborList

def updateDivBound(poi, current):
	for i in range(0, len(poi)):
		if poi[i] > current[i]:
			current[i] = poi[i]
	return current
	
def indexContruct(djgraph, nodeIndex, bounds, ldaDict, topicNum, nodeList, maxDist):
	index = {}
	g  = Graph(len(djgraph))
	
	edgeCount = 0


	for col in range(len(nodeList)):
		if djgraph[nodeIndex][col] == 0:
			continue
		else:
			direct = nodeList[col]
			
			index[direct] = {}
			'''
			for bound in bounds:
				index[direct][bound] = [0] * topicNum
			'''
				
			modG = [[0 for column in range(len(nodeList))] for row in range(len(nodeList))]
			for rowNum in range(len(modG)):
				if rowNum != nodeIndex:
					for colNum in range(len(modG[rowNum])):
						if colNum != nodeIndex:
							modG[rowNum][colNum] = djgraph[rowNum][colNum]
			#print modG[nodeIndex][col], djgraph[nodeIndex][col]
			#print modG[col][nodeIndex], djgraph[col][nodeIndex]
			modG[nodeIndex][col] = djgraph[nodeIndex][col]
			modG[col][nodeIndex] = djgraph[col][nodeIndex]
			#modifiedG[nodeIndex][col] = djgraph[nodeIndex][col]
			#modifiedG[col][nodeIndex] = djgraph[col][nodeIndex]
			
			#print modifiedG[nodeIndex][col], modifiedG[col][nodeIndex]
			
			g.graph = modG
 
			dist, perEdgeCount = g.dijkstra(nodeIndex)
			
			for i in range(len(dist)):
				if (dist[i] < maxDist) and (i != nodeIndex):
					index[direct][nodeList[i]] = dist[i]

			edgeCount = edgeCount + perEdgeCount
			
			#print dist
			'''
			for i in range(len(dist)):
				if dist[i] < sys.maxint:
					if i != nodeIndex:
						for bound in bounds:
							if dist[i] < bound:
								index[direct][bound] = updateDivBound(ldaDict[nodeList[i]], index[direct][bound])
								break
			'''
			
	
	return index, edgeCount
							

def main():
	#city = 'Tempe, Arizona, USA'
	#type = 'drive'
	max_distance = 2500
	bound = [500, 1000, 1500, 2000, max_distance]
	topicNum = 2
	if os.path.isfile('LargeEdge.csv') and os.path.isfile('LargeNode.csv'):
		print "Exist"
		network = []
		nodeSet = set()
		with open('LargeEdge.csv') as myfile:
			csv_reader = csv.reader(myfile, delimiter='|')
			for row in csv_reader:
				network.append([row[0], row[1], row[2]])
		with open('LargeNode.csv') as myfile:
			csv_reader = csv.reader(myfile, delimiter='|')
			for row in csv_reader:
				nodeSet.add(row[0])
	else:
		print "Download Map Now"
		network, nodeSet = getMap(city, type)
	
	print "Map Info"
	print len(nodeSet), len(network)

	neighborList = neighbor(network, nodeSet)
	
	nodeList = list(nodeSet)
	
	djGraph = [[0 for column in range(len(nodeList))] for row in range(len(nodeList))]
	
	for start, endPair in neighborList.items():
		row = nodeList.index(start)
		for end, length in endPair.items():
			col = nodeList.index(end)
			djGraph[row][col] = int(float(length))
		
	#print djGraph
	#print ldaDict
	#index = {}
	#distanceList = []
	'''
	for i in range(len(nodeList)):
		#singleIndex = indexContruct(djGraph, i, bound, ldaDict, topicNum, nodeList)
		#index[nodeList[i]] = singleIndex
		#print index[nodeList[i]]
		singleIndex = indexContruct(djGraph, i, bound, 'c', topicNum, nodeList)
		distanceList.append(singleIndex)
		print str(i) + '/' + str(len(nodeList))
		print singleIndex
	'''
	print "Start Building Index"

	with open('distInfo.csv', 'wb') as myfile:
		spamwriter = csv.writer(myfile)
		# Parallel Computing 
		# Allocate different range to different machines
		totalEdgeCount = 0
		start = timeit.default_timer();

		for i in range(len(nodeList)):
			#start = timeit.default_timer()
			singleIndex, singleEdgeCount = indexContruct(djGraph, i, bound, 'c', topicNum, nodeList, max_distance)
			#print singleEdgeCount
			spamwriter.writerow([nodeList[i], singleIndex])
			#stop = timeit.default_timer()
			#print singleIndex
			totalEdgeCount = totalEdgeCount + singleEdgeCount
			print str(i) + '/' + str(len(nodeList))

		stop = timeit.default_timer();

		print('Total Edge Count: ', totalEdgeCount)
		print('Total Time: ', stop - start);

	resFile = open('LargeRes.txt', 'w')
	resFile.write(totalEdgeCount)
	resFile.write(stop-start)
	resFile.close()


		#print('Time per node: ', (stop-start)/len(nodeList));
			##print('Time: ', stop - start)  
			
if __name__ == '__main__':
	main()