
import sys
import json
import re, string
import copy
from nltk.corpus import stopwords

regex = re.compile('[%s]' % re.escape(string.punctuation))
StopWord = stopwords.words('english')


class TweetClusering():
    def __init__(self, JsoneFile, InitialSeedsFile,iterations):

        tweets_ID = {}
        with open(JsoneFile, 'r') as f:
            for line in f:
                tweet = json.loads(line)
                tweets_ID[tweet['id']] = tweet

        f = open(InitialSeedsFile)
        InitialSeeds = [int(line.rstrip(',\n')) for line in f.readlines()]
        f.close()

        self.InitialSeeds = InitialSeeds
        self.tweets_ID = tweets_ID
        self.max_iterations = iterations
        self.k = len(InitialSeeds)
        self.clusters = {}  # cluster to tweetID
        self.rev_clusters = {}  # reverse index, tweetID to cluster
        self.JC_MAT = {}  # stores pairwise jaccard distance in a matrix

    def jaccardDistance(self, setA, setB):
        # Calcualtes the Jaccard Distance of two sets
        return 1 - float(len(setA.intersection(setB))) / float(len(setA.union(setB)))

    def Dictionary(self, string):
        # Returns a bag of words from a given string
        # Space delimited, removes punctuation, lowercase
        # Cleans text from url, stop words, tweet @, and 'rt'
        words = string.lower().strip().split(' ')
        for word in words:
            word = word.rstrip().lstrip()
            if not re.match(r'^https?:\/\/.*[\r\n]*', word) \
                    and not re.match('^@.*', word) \
                    and not re.match('\s', word) \
                    and word not in StopWord \
                    and word != 'rt' \
                    and word != '':
                yield regex.sub('', word)

    def Create_Matrix(self):
        # Dynamic Programming: creates matrix storing pairwise jaccard distances
        for ID1 in self.tweets_ID:
            self.JC_MAT[ID1] = {}
            TW1 = set(self.Dictionary(self.tweets_ID[ID1]['text']))
            for ID2 in self.tweets_ID:
                if ID2 not in self.JC_MAT:
                    self.JC_MAT[ID2] = {}
                TW2 = set(self.Dictionary(self.tweets_ID[ID2]['text']))
                distance = self.jaccardDistance(TW1, TW2)
                self.JC_MAT[ID1][ID2] = distance
                self.JC_MAT[ID2][ID1] = distance

    def Create_Clusters(self):
        # Initialize tweets_ID to no cluster
        for ID in self.tweets_ID:
            self.rev_clusters[ID] = -1

        # Initialize clusters with InitialSeeds
        for k in range(self.k):
            self.clusters[k] = set([self.InitialSeeds[k]])
            self.rev_clusters[self.InitialSeeds[k]] = k

    def calcNewClusters(self):
        # Initialize new cluster
        new_clusters = {}
        new_rev_cluster = {}
        for k in range(self.k):
            new_clusters[k] = set()

        for ID in self.tweets_ID:
            min_dist = float("inf")
            min_cluster = self.rev_clusters[ID]

            # Calculate min average distance to each cluster
            for k in self.clusters:
                dist = 0
                count = 0
                for ID2 in self.clusters[k]:
                    dist += self.JC_MAT[ID][ID2]
                    count += 1
                if count > 0:
                    avg_dist = dist / float(count)
                    if min_dist > avg_dist:
                        min_dist = avg_dist
                        min_cluster = k
            new_clusters[min_cluster].add(ID)
            new_rev_cluster[ID] = min_cluster
        return new_clusters, new_rev_cluster

    def optimize(self):
        # Initialize previous cluster to compare changes with new clustering
        new_clusters, new_rev_clusters = self.calcNewClusters()
        self.clusters = copy.deepcopy(new_clusters)
        self.rev_clusters = copy.deepcopy(new_rev_clusters)

        # Converges until old and new iterations are the same
        iterations = 1
        while iterations < self.max_iterations:
            new_clusters, new_rev_clusters = self.calcNewClusters()
            iterations += 1
            if self.rev_clusters != new_rev_clusters:
                self.clusters = copy.deepcopy(new_clusters)
                self.rev_clusters = copy.deepcopy(new_rev_clusters)
            else:
                return

    def calc_SSE(self):
        self.SSE = []
        for cluster in self.clusters:
            error = 0
            for ID1 in self.clusters[cluster]:
                error += (self.JC_MAT[self.InitialSeeds[cluster]][ID1])
            self.SSE.append(error)

    def printClusters(self,out_file):
        # Prints cluster ID and tweet IDs for that cluster
        file = open(out_file, 'w+')
        file.write('CS6375.004 - Machine Learning: Assignment-6 : Part II - Tweets Clustering using k-means \n\n')
        K_clusters=[]
        for k in self.clusters:
            K_clusters.append(str(k+1) + ':' + ','.join(map(str, self.clusters[k])))
            #print(str(k+1) + ' : ' + ','.join(map(str, self.clusters[k])))
            file.write('Cluster '+ str(k+1) + ' : ' + ', '.join(map(str, self.clusters[k])))
            file.write("\n")

        '''
        print("======================================================")
        for cluster in self.clusters:
            print('For cluster '+str(cluster+1) + ' : SSE = ' + str(round(self.SSE[cluster], 3)))
        '''
        TotalSSE = sum(self.SSE)
        print("======================================================")
        print("Validation SSE = %3.4f"%TotalSSE)
        print('\nResults are available in output file')
        file.write("======================================================")
        file.write("\n")
        file.write("Number of Clusters = %d" % len(self.SSE))
        file.write("\n")
        file.write("Validation SSE = %3.4f" % TotalSSE)

# the main part starts from here
if len(sys.argv) != 4:
    print("Error : Please enter files in this format <json file> <InitialSeeds file> <outputfile.txt>")
    exit(-1)
# initialize code and load data
TC = TweetClusering(sys.argv[1], sys.argv[2],1000)
# Create clusters based on initial seeds
TC.Create_Clusters()
# create distance matrix using Jaccard distance
TC.Create_Matrix()
# optimize to find the best clusters
TC.optimize()
# calculate SSE
TC.calc_SSE()
# show and save results in the given text files
TC.printClusters(sys.argv[3])

