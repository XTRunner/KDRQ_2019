import json
import csv
from random import randint
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import math
import cPickle
import os.path
import pickle


class nodelog:
    def __init__(self, poiNo, offset):
        self.poiNo = poiNo
        self.offset = offset

class direction:
    def __init__(self, direct, dist):
        self.direct = direct
        self.dist = dist

class bucket:
    def __init__(self, nodes, upperBound):
        self.nodes = nodes
        self.upperBound = upperBound

class Graph():
    
    # Graph as two-dimension list
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] for row in range(vertices)]

    # Find the next node with minimum distance and not included into sptSet
    # If find, return index. If not, then rest of nodes are unconnnected
    def minDistance(self, dist, sptSet, optSet):
        
        min = 2000
        # Search not nearest vertex not in the 
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
        
        flag = False
        
        if min_index in optSet:
            optSet.remove(min_index)
            flag = True
                    
        if optSet:
            return min_index, flag, 'continue', optSet
        else:
            return min_index, flag, 'DONE', optSet

    def dijkstra(self, src, optList, poiDict, nodeList):
        # dist: [0, infinity, infinity,...]
        dist = [sys.maxint] * self.V
        dist[src] = 0
        
        #res = {}

        finalRes = []
        
        optSet = set(optList)
        if src in optSet:
            optSet.remove(src)
            # curRes = []
            finalRes = greedy(finalRes, nodeList[src], poiDict)
            res = [(0, sum(updateUtil(finalRes, poiDict)))]
        else:
            res = [(0,0)]
        # sptSet: [F,F,F,...]
        sptSet = [False] * self.V
        #edgeCount = 0
        start_time = time.time()
        for cout in range(self.V):
            u, flag, done, optSet = self.minDistance(dist, sptSet, optSet)
            stop_time = time.time()
            # Put the minimum distance vertex in the 
            # shotest path tree
            if flag and done == 'DONE':
                finalRes = greedy(finalRes, nodeList[u], poiDict)

                #res[edgeCount] = u
                res.append((stop_time-start_time, sum(updateUtil(finalRes, poiDict))))
                break
            elif (flag == False) and done ==  'DONE':
                res.append((stop_time-start_time, sum(updateUtil(finalRes, poiDict))))
                break
            else:
                sptSet[u] = True            
                # node has not included in sptSet and current value is greater than new distance
                for v in range(self.V):
                    if self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]:
                        dist[v] = dist[u] + self.graph[u][v]
                if flag:
                    #res[edgeCount] = u
                    finalRes = greedy(finalRes, nodeList[u], poiDict)
                    res.append((stop_time-start_time, sum(updateUtil(finalRes, poiDict))))
            #edgeCount += 1
                    
        #print dist
        return res

def poicheck():
    poiDict = {}
    with open('matchPoIlda.csv') as rfile:
        csvreader = csv.reader(rfile)
        for row in csvreader:
            node = row[0]
            div = row[1].replace("[","")
            div = div.replace("]","")
            div = div.split(",")
            for i in range(len(div)):
                div[i] = round(float(div[i]), 1)
            if node in poiDict.keys():
                poiDict[node].append(div)
            else:
                poiDict[node] = [div]

    output = open('matchPoIlda.pkl', 'wb')

    pickle.dump(poiDict, output)

    output.close()

    return poiDict

def updateDivBound(poi, current):
    for i in range(len(poi)):
        if poi[i] > current[i]:
            current[i] = poi[i]
    return current

def indexConstruct(poiDict, boundary, topicNum = 10):
    data = {}
    '''
    poiDict
    {
    ...
    nodei: [[diveristy1], [diveristy2], ...]
    ...
    }
    '''
    # All the poi node
    poiList = poiDict.keys()
    bounds = boundary
    '''
    data
    {
    ...
    nodei: { ( nodej, dist(i,j) ): {500: ([PoI1, PoI2, ...], [upper bound of diveristy]),
                                    1000: ...
                                    ...
                                    }
            
             ( nodek, dist(i,k) ): {500: ([PoI3, PoI4, ...], [upper bound of diveristy]),
                                    1000: ...
                                    ...
                                    }
             ...
             }
    ...
    }
    '''
    with open('distInfo.csv') as rfile:
        csvreader = csv.reader(rfile)
        countRowNum = 0
        for row in csvreader:
            countRowNum += 1
            # start: query point
            start = row[0]
            data[start] = {}
            # Just for using json toolbox
            jsonFile = json.loads(str(row[1]).replace("'", '"'))
            # direct: Direct node
            for direct in jsonFile:
                # If not {}, which means that even neighbor node further than distance range
                if jsonFile[direct]:
                    # neigh: ( nodej, dist(i,j) )
                    neigh = direction(direct, jsonFile[direct][direct])
                    data[start][neigh] = {}
                    # For each boundaries
                    for bound in bounds:
                        # Init [], [0,0,0,...]
                        data[start][neigh][bound] = bucket([], [0]*topicNum)
                    # key: touchable node through direct
                    for key in jsonFile[direct]:
                        if key in poiList:
                            # Check which bucket
                            for bound in bounds:
                                if jsonFile[direct][key] < bound:
                                    data[start][neigh][bound].nodes.append(key)
                                    for diveristy in poiDict[key]:
                                        data[start][neigh][bound].upperBound = updateDivBound(diveristy, data[start][neigh][bound].upperBound)
                                    break   
                    #print data[start][neigh][bound].nodes
                    #print data[start][neigh][bound].upperBound
            print str(countRowNum) + "/4604"

    if len(bounds) > 5:
        filName = 'data2.pkl'
    elif 1732 in bounds:
        filName = 'data3.pkl'
    else:
        filName = 'data1.pkl'

    output = open(filName, 'wb')

    pickle.dump(data, output)

    output.close()

    return data

def updateUtil(poiList, poiDict, topicNum = 10):
    divRes = []
    #print poiList
    for i in range(topicNum):
        tmp = 1
        for poi in poiList:
            tmp = tmp * (1 - poiDict[poi.poiNo][poi.offset][i])
        finalTmp = 1 - tmp
        divRes.append(finalTmp)
    return divRes

def utility(current, direct, drange, used, bounds, theta, topicNum = 10):
    res = 0
    remain = drange - used
    #print remain
    if remain >= 0:
        for i in range(topicNum):
            res += direct[bounds[0]].upperBound[i] * (1-current[i])
        for level in range(len(bounds)):
            if remain > bounds[level]:
                for i in range(topicNum):
                    res += direct[bounds[level+1]].upperBound[i] * (1-current[i]) * (theta**(level+1))
        return res
    else:
        return 0
    
def greedy(poiList, newPoI, poiDict, k=5):
    unionIndex = []
    for poi in poiList:
        unionIndex.append(poi)
    for i in range(len(poiDict[newPoI])):
        # union.append(poiDict[newPoI][i])
        unionIndex.append(nodelog(newPoI, i))
    res = []
    length = len(unionIndex)
    tmpList = []
    maxi = 0

    if length > k:
        for a in range(length - 4):
            for b in range(a + 1, length - 3):
                for i in range(b + 1, length - 2):
                    for j in range(i + 1, length - 1):
                        for k in range(j + 1, length):
                            tmpList = []
                            tmpList.append(unionIndex[a])
                            tmpList.append(unionIndex[b])
                            tmpList.append(unionIndex[i])
                            tmpList.append(unionIndex[j])
                            tmpList.append(unionIndex[k])
                            if sum(updateUtil(tmpList, poiDict)) > maxi:
                                maxi = sum(updateUtil(tmpList, poiDict))
                                res = []
                                res.append(unionIndex[a])
                                res.append(unionIndex[b])
                                res.append(unionIndex[i])
                                res.append(unionIndex[j])
                                res.append(unionIndex[k])
        return res
    else:
        return unionIndex

def findNeighDist(data, node1, node2):
    passNode = node1.split(";")
    for i, j in data[passNode[len(passNode)-1]].items():
        if i.direct == node2:
            jidistance = i.dist
            break
    return jidistance

def search(queryP, data, poiDict, theta, advancedFlag, topicNum = 10, drange = 2000, k = 3):
    
    if advancedFlag == 0:
        bounds = [1000,2000]
        bounds = bounds[:(bounds.index(drange)+1)]
    elif advancedFlag == 1:
        bounds = [500, 1000, 1500, 2000]
        bounds = bounds[:(bounds.index(drange)+1)]
    elif advancedFlag == 2:
        bounds = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
        bounds = bounds[:(bounds.index(drange)+1)]
    elif advancedFlag == 3:
        bounds = [1000, 1414, 1732, 2000]
        bounds = bounds[:(bounds.index(drange)+1)]
    # Current upper bound of diveristy
    divRes = [0]* topicNum
    finalRes = []
    # All PoIs within distance range
    searchQueue = set()
    utilDict = {}
    
    #log = []

    #addIn = []
    #kickOut = []

    for direction1, buckets in data[queryP].items():
        for bound in bounds:
            if drange >= bound:
                for node in buckets[bound].nodes:
                    searchQueue.add(node)
    
    PoINum = len(searchQueue)

    resSearchQueue = []

    for poiLoc in searchQueue:
        for i in range(len(poiDict[poiLoc])):
            resSearchQueue.append((poiLoc, i))
    # initial first entry in utilDict
    utilDict[queryP] = {}
        
    edgeCount = 0
    
    divList = []
    
    if queryP in searchQueue:
        searchQueue.remove(queryP)
        #curRes = []
        finalRes = greedy(finalRes, queryP, poiDict)
        divRes = updateUtil(finalRes, poiDict)
        divList.append((0, sum(divRes)))
    else:
        divList.append((0,0))
    
    for direction1, buckets in data[queryP].items():
        utilDict[queryP][direction1.direct] =  direction(utility(divRes, buckets, drange, 0, bounds, theta), 0)
        #print prone(buckets, drange, 0)
    
    start_time = time.time()
    while searchQueue:
        
        maxDirect = None
        nextStep = None

        for start, ends in utilDict.items():
            if not ends:
                utilDict.pop(start)
            else:
                for end, value in ends.items():
                    if maxDirect and value.direct > maxDirect:
                        maxDirect = value.direct
                        nextStep = [start, end]
                    elif not maxDirect:
                        maxDirect = value.direct
                        nextStep = [start, end] 
        
        edgeCount += 1
        
        #print nextStep[0], nextStep[1], maxDirect
        #print nextStep, maxDirect
        # If next step is a PoI
        if nextStep[1] in searchQueue:
            #curRes = []

            #for i in finalRes:
            #   curRes.append(i)

            #maxScore = sum(updateUtil(curRes, poiDict))
            # multiple PoIs in one node
            stop_time = time.time()
            finalRes = greedy(finalRes, nextStep[1], poiDict)
            #log.append([edgeCount, [[i.poiNo, i.offset] for i in addIn], [[i.poiNo, i.offset] for i in kickOut]])
            
            #print kickOut

            divRes = updateUtil(finalRes, poiDict)
            divList.append((stop_time - start_time, sum(divRes)))

            searchQueue.remove(nextStep[1])
            
            # divRes changed, update utility of existed index
            for start, entries in utilDict.items():
                for direct2, entry in entries.items():
                    for foundKey, foundVal in data[start.split(";")[-1]].items():
                        if foundKey.direct == direct2:
                            utilDict[start].pop(direct2)
                            utilDict[start][direct2] = direction(utility(divRes, foundVal, drange, entry.dist, bounds, theta), entry.dist)
                            break
        
        #print nextStep[0], nextStep[1]
        #print searchQueue
        #divList.append(sum(divRes))
        #print searchQueue
        # Remove start -> neighbor from utilDict
        utilDict[nextStep[0] + ';' + nextStep[1]] = {}
        
        passNode = nextStep[0].split(";")
        
        for i, j in data[passNode[len(passNode)-1]].items():
            if i.direct == nextStep[1]:
                jidistance = i.dist
                break
        
        shortestUsed = utilDict[nextStep[0]][nextStep[1]].dist + jidistance
        #print shortestUsed
        # Insert new node and new directions into index
        for direction1, buckets in data[nextStep[1]].items():
            #print direction1.direct
            if direction1.direct != nextStep[1] and (direction1.direct not in passNode) and shortestUsed <= drange:
                utilDict[nextStep[0] + ';' + nextStep[1]][direction1.direct] =  direction(utility(divRes, buckets, drange, shortestUsed, bounds, theta), shortestUsed)
        
        #print utilDict
        utilDict[nextStep[0]].pop(nextStep[1])
        #print utilDict
        #print '////////////////////////'
        
        for filterDirect in utilDict[nextStep[0] + ';' + nextStep[1]].keys():
            minDist = None
            for startNode, nextNodes in utilDict.items():
                for nextNode, nextInfo in nextNodes.items():
                    if nextNode == filterDirect:
                        if not minDist:
                            minDist = nextInfo.dist + findNeighDist(data, startNode, nextNode)
                            minIndex = [startNode, nextNode]
                        elif minDist and (nextInfo.dist + findNeighDist(data, startNode, nextNode)) >= minDist:
                            utilDict[startNode].pop(nextNode)
                        elif minDist and (nextInfo.dist + findNeighDist(data, startNode, nextNode)) < minDist:
                            minDist = nextInfo.dist
                            utilDict[minIndex[0]].pop(minIndex[1])
                            minIndex = [startNode, nextNode]
        
        #print utilDict
        #print '.......................'
        #print ele.direct
        
        #print utilDict
        
    return finalRes, divList
    
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
    
def djALG(djgraph, i, poiList, poiDict, nodeList):

    g  = Graph(len(djgraph))
    
    g.graph = djgraph

    #start = time.time()
    res = g.dijkstra(i, poiList, poiDict, nodeList)
    #stop = time.time()
    return res

def baseLine(data, poiDict, drange=2000):
    
    bounds = [500, 1000, 1500, 2000]
    bounds = bounds[:(bounds.index(drange)+1)]

    network = []
    nodeSet = set()
    with open('Tempe, Arizona, USAdriveedge.csv') as myfile:
        csv_reader = csv.reader(myfile, delimiter='|')
        for row in csv_reader:
            network.append([row[0], row[1], row[2]])
    with open('Tempe, Arizona, USAdrivenode.csv') as myfile:
        csv_reader = csv.reader(myfile, delimiter='|')
        for row in csv_reader:
            nodeSet.add(row[0])
    
    print "BaseLine"

    neighborList = neighbor(network, nodeSet)
    
    nodeList = list(nodeSet)
    
    djGraph = [[0 for column in range(len(nodeList))] for row in range(len(nodeList))]
    
    for start, endPair in neighborList.items():
        row = nodeList.index(start)
        for end, length in endPair.items():
            col = nodeList.index(end)
            djGraph[row][col] = int(float(length))
    
    #resultDiversity = {}
    
    djAnswer = []
    #avgTime = 0
    
    for i in range(200):
        
        searchQueue = set()
        
        for direction1, buckets in data[nodeList[i]].items():
            for bound in bounds:
                if drange >= bound:
                    for node in buckets[bound].nodes:
                        searchQueue.add(node)
        
        searchList = list(searchQueue)
        
        poiList = []
        
        for entry in searchList:
            poiList.append(nodeList.index(entry))
        
        #start = time.time()
        singleDiversity = djALG(djGraph, i, poiList, poiDict, nodeList)
        #print singleDiversity

        djAnswer.append(singleDiversity)
        #stop = time.time()
        
        #singleTime = stop -start
        
        #print ("total", singleTime)
        
        '''
        finalDiversity = []
        
        maxLen = max(singleDiversity.keys())
        
        #avgTime += singleTime
        
        currentUtility = 0
        
        poiList = []
        
        for j in range(maxLen):
            if not j in singleDiversity.keys():
                finalDiversity.append(currentUtility)
            else:
                poiList = greedy(poiList, poiDict[nodeList[singleDiversity[j]]])
                currentUtility = sum(updateUtil(poiList))
                finalDiversity.append(currentUtility)
        '''
        
        #print str(i+1) + '/200'
        
        #resultDiversity[nodeList[i]] = finalDiversity
        
    
    #print ('Time for DJ algorithm:', avgTime)

    return djAnswer
    
    
def main():
    
    '''
    poiDict
    {
    ...
    nodei: [[diveristy1], [diveristy2], ...]
    ...
    }
    '''
    '''
    data
    {
    ...
    nodei: { ( nodej, dist(i,j) ): {500: ([PoI1, PoI2, ...], [upper bound of diveristy]),
                                    1000: ...
                                    ...
                                    }
            
             ( nodek, dist(i,k) ): {500: ([PoI3, PoI4, ...], [upper bound of diveristy]),
                                    1000: ...
                                    ...
                                    }
             ...
             }
    ...
    }
    '''
    if os.path.isfile('matchPoIlda.pkl'):
        pkl_file = open('matchPoIlda.pkl', 'rb')
        poiDict = pickle.load(pkl_file)
        pkl_file.close()
    else:
        poiDict = poicheck()
    
    bounds1 = [500, 1000, 1500, 2000]
    #bounds2 = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    #bounds3 = [1000, 1414, 1732, 2000]
    
    if os.path.isfile('data1.pkl'):
        pkl_file = open('data1.pkl', 'rb')
        data1 = pickle.load(pkl_file)
        pkl_file.close()
    else:
        data1 = indexConstruct(poiDict, bounds1)
        #data2 = indexConstruct(poiDict, bounds2)
        #data3 = indexConstruct(poiDict, bounds3)
        # nodeList: all the nodes in road network

    nodeSet = set()
    with open('Tempe, Arizona, USAdrivenode.csv') as myfile:
        csv_reader = csv.reader(myfile, delimiter='|')
        for row in csv_reader:
            nodeSet.add(row[0])

    nodeList = list(nodeSet)
    saveRes = []
    for i in range(200):
        finalRes, divList = search(nodeList[i], data1, poiDict, 0, 1, drange=2000)
        #print divList
        saveRes.append(divList)

    with open("SSTD_res/index_lambda0_bound1_k5_time", "wb") as output_file:
        cPickle.dump(saveRes, output_file)


    '''
    djRes = baseLine(data1, poiDict, drange=2000)
    with open("SSTD_res/dijkstra_k3_time", "wb") as output_file:
        cPickle.dump(djRes, output_file)
    #print djRes

    '''
    '''
    resDJ = []
    maxLenDJ = 0
    lenDJList = []
    for nodeID, divList in djRes.items():
        if divList and divList[-1] > 0:
             if len(divList) > maxLenDJ:
                maxLenDJ = len(divList)
             resDJ.append(divList)
             lenDJList.append(len(divList))
             
    for entry in resDJ:
        if len(entry) < maxLenDJ:
            for i in range(maxLenDJ-len(entry)):
                entry.append(entry[-1])

    
    finalPlotDJ =[]
    lenDJ = len(resDJ)
    for i in range(maxLenDJ):
        sum = 0
        normSum = 0
        for entry in resDJ:
            sum += entry[i]
        finalPlotDJ.append(float(sum)/lenDJ)
    
    with open('dijkstraResk3.csv', 'a') as wfile:
        spamwriter = csv.writer(wfile)
        for entry in finalPlotDJ:
            spamwriter.writerow([entry])

    #nodeList = data1.keys()
    '''


    #updateStore = {}
    #resLogOutput = {}

    #finalRes, divList, resSearchQueue, log = search('317675675', data1, poiDict, 0, 1, drange = 2000)

    #print [(i.poiNo, i.offset) for i in finalRes], log
    '''

    for i in range(200):
        finalRes, divList, resSearchQueue, log = search(nodeList[i], data1, poiDict, 0, 1, drange = 2000)
        updateStore[nodeList[i]] = [[(j.poiNo, j.offset) for j in finalRes], resSearchQueue]
        resLogOutput[nodeList[i]] = log

    output1 = open('plainRes.pkl', 'wb')
    pickle.dump(updateStore, output1)
    output1.close()

    output2 = open('resLog.pkl', 'wb')
    pickle.dump(resLogOutput, output2)
    output2.close()
    '''



    '''
    djRes = baseLine(data1, poiDict, drange=2000)
    
    resDJ = []
    maxLenDJ = 0
    lenDJList = []
    for nodeID, divList in djRes.items():
        if divList and divList[-1] > 0:
             if len(divList) > maxLenDJ:
                maxLenDJ = len(divList)
             resDJ.append(divList)
             lenDJList.append(len(divList))
             
    for entry in resDJ:
        if len(entry) < maxLenDJ:
            for i in range(maxLenDJ-len(entry)):
                entry.append(entry[-1])

    
    finalPlotDJ =[]
    lenDJ = len(resDJ)
    for i in range(maxLenDJ):
        sum = 0
        normSum = 0
        for entry in resDJ:
            sum += entry[i]
        finalPlotDJ.append(float(sum)/lenDJ)
    
    with open('dijkstraResk3.csv', 'a') as wfile:
        spamwriter = csv.writer(wfile)
        for entry in finalPlotDJ:
            spamwriter.writerow([entry])
    
    nodeCom = djRes.keys()
    
    res01 = []
    res11 = []

    maxLen01 = 0
    maxLen11 = 0
    
    len01List = []
    len11List = []
    

    #start = time.time()
    for i in nodeCom:
                
        divList, PoINum = search(i, data1, poiDict, 0, 1, drange = 2000)
        if divList and divList[-1] > 0: 
            res01.append(divList)
            if len(divList) > maxLen01:
                maxLen01 = len(divList)
                
        divList, PoINum = search(i, data1, poiDict, 1, 1, drange = 2000)
        if divList and divList[-1] > 0: 
            res11.append(divList)
            if len(divList) > maxLen11:
                maxLen11 = len(divList) 
        
        print str(nodeCom.index(i)+1) + '/200'
    
    for entry in res01:
        if len(entry) < maxLen01:
            for i in range(maxLen01-len(entry)):
                entry.append(entry[-1])
                
    for entry in res11:
        if len(entry) < maxLen11:
            for i in range(maxLen11-len(entry)):
                entry.append(entry[-1])
    
    finalPlot01 =[]
    len1 = len(res01)
    for i in range(maxLen01):
        sum = 0
        for entry in res01:
            sum += entry[i]
        finalPlot01.append(float(sum)/len1)
    
    finalPlot11 =[]
    len1 = len(res11)
    for i in range(maxLen11):
        sum = 0
        for entry in res11:
            sum += entry[i]
        finalPlot11.append(float(sum)/len1)
    
    with open('theta0k3.csv', 'a') as wfile:
        spamwriter = csv.writer(wfile)
        for entry in finalPlot01:
            spamwriter.writerow([entry])
            
    with open('theta1k3.csv', 'a') as wfile:
        spamwriter = csv.writer(wfile)
        for entry in finalPlot11:
            spamwriter.writerow([entry])
    
    plt.plot(finalPlot01, linestyle='-', color='black', linewidth=2, label = "theta = 0")
    plt.plot(finalPlot11, linestyle='-.', color='blue', linewidth=2, label = "theta = 1")
    plt.plot(finalPlotDJ, linestyle='--', color='red', linewidth=2, label = "Dijkstra Algorithm")
    plt.legend(loc='best')
    plt.axis([0, 400, 0, 3.5])
    plt.title("Comparsion of different algorithms with k = 3")
    plt.show()
    
    logX = []
    
    for x in range(0,401):
        logX.append(float(math.sqrt(x)))
    
    plt.plot(logX, finalPlot01[:401], linestyle='-', color='black', linewidth=2, label = "theta = 0")
    plt.plot(logX, finalPlot11[:401], linestyle='-.', color='blue', linewidth=2, label = "theta = 1")
    plt.plot(logX, finalPlotDJ[:401], linestyle='--', color='red', linewidth=2, label = "Dijkstra Algorithm")
    plt.legend(loc='best')
    plt.axis([0, 18, 0, 3.5])
    plt.title("Comparsion of different algorithms with k = 3")

    plt.show()

    '''

if __name__ == '__main__':
    main()