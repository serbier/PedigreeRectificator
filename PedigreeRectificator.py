import argparse
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from matplotlib.patches import Circle, Wedge, Polygon, Rectangle

class PedigreeRectificator:

    def __init__(self,df,workDir = './',**data):
        self.mother = None
        self.father = None
        self.child = None
        self.args = data
        self.workDir = workDir
        self.chromosomeSizes = rec_dd()
        self.gdf = df
        self.df = pd.DataFrame()
        self.getFiltredDf()
        self.genDict = self.dataSegmentation()
        self.setChromosomeRange()
        self.family = set(data.keys())
        self.getRectificationCase()
        self.heterozygous = list()
        self.unknownOrigin = list()
        self.lostOrigin = list()
        self.motherOrigin = list()
        self.fatherOrigin = list()
        self.mkMother = None 
        self.mkFather = None
        self.mkLost = None
        self.mkUnknown = None
        self.mktotal = None
        self.mkMonomorphic = None
        self.motherMarkers = None
        self.fatherMarkers = None
        self.unknownMarkers = None
        self.lostMarkers = None
        self.getMarkerAncestry()
        self.chrGuide = chrGuide = [
        "Chr01",
        "Chr02",
        "Chr03",
        "Chr04",
        "Chr05",
        "Chr06",
        "Chr07",
        "Chr08",
        "Chr09",
        "Chr10",
        "Chr11"
        ]
        
    def getFiltredDf(self):
        notIndNames = ['POS','CHR','Id', 'Reference', 'Alternative']
        pedigreeArgs = ['p1', 'p2', 'p11', 'p12', 'p21', 'p22', 'child']
        for i in pedigreeArgs:
            name = self.args[i]
            if name != None:
                notIndNames.append(name)
        self.df = self.gdf[notIndNames]
        try:
            self.df['PRED'] = self.gdf['PRED']
        except KeyError:
            pass

        self.df = self.df.dropna()  
    
    def getRectificationCase(self):
        if set(['p1', 'p2']).issubset(self.family):
            mother = self.genDict[self.args['p1']]
            father = self.genDict[self.args['p2']]
            child = self.genDict[self.args['child']]
            filtrMks = getInformativeMarkers(mother, father, child)

        elif set(['p1', 'p21', 'p22']).issubset(self.family):
            mother = self.genDict[self.args['p1']]
            father = getExpectedGenotype( self.genDict[self.args['p21']], self.genDict[self.args['p22']])
            child = self.genDict[self.args['child']]
            filtrMks = getInformativeMarkers(mother, father, child)

        elif set(['p2', 'p11', 'p12']).issubset(self.family):
            father = self.genDict[self.args['p2']]
            mother = getExpectedGenotype( self.genDict[self.args['p11']], self.genDict[self.args['p12']])
            child = self.genDict[self.args['child']]
            filtrMks = getInformativeMarkers(mother, father, child)
        elif set(['p21', 'p22', 'p11', 'p12']).issubset(self.family):
            father = getExpectedGenotype( self.genDict[self.args['p21']], self.genDict[self.args['p22']])
            mother = getExpectedGenotype( self.genDict[self.args['p11']], self.genDict[self.args['p12']])
            child = self.genDict[self.args['child']]
            filtrMks = getInformativeMarkers(mother, father, child)
        else:
            return False
        self.mother = filtrMks['p1'] 
        self.father = filtrMks['p2']
        self.child  = filtrMks['child'] 


    def getMarkerAncestry(self): 

        for ch in self.child.keys():
            for pos in self.child[ch].keys():
                locusChild = self.child[ch][pos]
                allelesF = self.father[ch][pos]
                allelesM = self.mother[ch][pos]
                ancestryAlleles = allelesF.union(allelesM)
                if len(locusChild) > 1:
                    if len(locusChild.difference(ancestryAlleles)) > 0:
                        self.unknownOrigin.append(pos)

                    if allelesF != allelesM:
                        if locusChild == allelesF:
                            self.lostOrigin.append(pos)
                        elif locusChild == allelesM:
                            self.lostOrigin.append(pos)

                    self.heterozygous.append(pos)
                elif len(locusChild) == 1:

                    if allelesF == locusChild and allelesF != allelesM:
                        self.fatherOrigin.append(pos)
                    elif allelesM == locusChild and allelesM != allelesF:
                        self.motherOrigin.append(pos)

                    else :
                        if locusChild.issubset(allelesF) and locusChild.issubset(allelesM):
                            self.lostOrigin.append(pos)
                        elif locusChild.issubset(allelesF):
                            self.fatherOrigin.append(pos)
                        elif locusChild.issubset(allelesM):
                            self.motherOrigin.append(pos)
                        else:
                            self.unknownOrigin.append(pos)

        self.mkMother = len(self.motherOrigin)
        self.mkFather = len(self.fatherOrigin)
        self.mkLost = len(self.lostOrigin)
        self.mkUnknown = len(self.unknownOrigin)
        self.mktotal = self.df.shape[0]
        self.mkMonomorphic = self.mktotal - self.mkMother - self.mkFather - self.mkLost - self.mkUnknown

        print("Mother Markers: %s"%(self.mkMother))
        print("Father Markers: %s"%(self.mkFather))
        print("Monomorphic Markers: %s"%(self.mkMonomorphic))
        print("Lost Origin Markers: %s"%(self.mkLost))
        print("Unknown Markers: %s"%(self.mkUnknown))
        print("Total Markers: %s"%(self.mktotal))


        self.motherMarkers =self.df.loc[self.df["POS"].isin(self.motherOrigin)] ,
        self.fatherMarkers =self.df.loc[self.df["POS"].isin(self.fatherOrigin)] ,
        self.unknownMarkers =self.df.loc[self.df["POS"].isin(self.unknownOrigin)],
        self.lostMarkers =self.df.loc[self.df["POS"].isin(self.lostOrigin)] 




    def dataSegmentation(self):
        columns = self.df.columns
        notIndNames = ['POS', 'PRED', 'CHR','Id', 'Reference', 'Alternative']
        indNames = [i for i in columns if i not in notIndNames]
        segmentedData = rec_dd()
        duplicateData = self.df[self.df.duplicated('POS', keep=False)]
        if duplicateData.shape[0] > 0:
            duplicateData.to_csv(self.workDir+"/duplMarkers.csv"%(), sep="\t")

        df.drop_duplicates('POS', inplace=True)

        for n,row in self.df.iterrows():
            for indName in indNames:
                if len(row[indName]) > 0:
                    segmentedData[indName][row['CHR']][row['POS']] = set(row[indName].split("/")) 

        return segmentedData
    
    
    

    def getChrPossibleIntrogressionBlocks(self,df,distance_threshold,distanceType):
        df = df[pd.notnull(df[distanceType])]
        chro = df.CHR.unique()
        X = np.array(df[distanceType]).reshape(-1,1)
        clustering = AgglomerativeClustering(affinity='euclidean', compute_full_tree=True,
                            connectivity=None, distance_threshold=distance_threshold,
                            linkage='single', memory=None, n_clusters=None,
                            pooling_func='deprecated')
        clustering.fit(X)
        df.insert(loc=0, column='cluster', value=clustering.labels_)
        out = pd.DataFrame(columns=["CHR","No_Introgresssion", "Min_Bound", "Max_Bound", "No_Markers", "Length"])
        #print("Int No.\t Min Bnd\tMax Bnd \tNo. \tLength")
        for n,posIntro in df.groupby('cluster'):
            minBound = posIntro.POS.min()
            maxBound = posIntro.POS.max()
            markerCount = posIntro.POS.count()
            length = maxBound-minBound
            out.loc[n] = [chro[0],n+1, minBound, maxBound, markerCount, length]
            #print("%s\t %s\t %s\t %s\t %s pb"%(n+1, minBound, maxBound, markerCount, length))
        return(out.sort_values('Min_Bound'))
    def getPossibleGenomeIntroBlocks(self,df,distanceType='PRED',distance_threshold = 0.5):
    
        notGp = pd.DataFrame()
        gp = pd.DataFrame()
        for chromosome,chrDf in df.groupby('CHR'):
            try:
                
                gp = gp.append(self.getChrPossibleIntrogressionBlocks(chrDf, distance_threshold,distanceType))

            except ValueError:
                notGp.append(chrDf)
        return [gp, notGp]
    
    def setChromosomeRange(self):
        
        for chromosome in self.df.CHR.unique():
            self.chromosomeSizes[chromosome] = self.df[self.df['CHR'] == chromosome]['POS'].max()
    
    def plotRegions(self, df, distance_threshold = 0.5, distanceType = 'PRED',margin = 1.0, width = 0.4, textPad = 1.1):
        df = self.getPossibleGenomeIntroBlocks(df,distanceType = distanceType,distance_threshold=distance_threshold)[0]
        fig = plt.figure(figsize=(20,11))
        ax = fig.add_subplot(111)
        pad = 1
        maxlim = getMaxDicValue(self.chromosomeSizes)
        counter = margin
        tickPos = list()
        regions = df
        Rcolor = "#D46A6A"
        Bcolor = "#7CC0C3"
        for chromosome in self.chrGuide:
            maxLim = np.interp(self.chromosomeSizes[chromosome], (0, maxlim), (0, 10))
            r = Rectangle((counter, 0), width, maxLim,color = Bcolor, alpha= 0.5)
            cap = Wedge((counter+width/2.0,maxLim), width/2.0,0, 180,color = Bcolor, alpha=0.5, linewidth = 0)
            bcap = Wedge((counter+width/2.0,0), width/2.0,180,0,color = Bcolor, alpha=0.5)
            xcenter = counter+width/2.0    
            tickPos.append(xcenter)
            ax.add_patch(r)
            ax.add_patch(bcap)
            ax.add_patch(cap)
            chRegions = regions[regions['CHR'] == chromosome]
            count = 0 
            for n,row in chRegions.iterrows():
                if row['Length'] >= 0:
                    formattedLength = "{:,}".format(row["Length"])
                    minBound = np.interp(row['Min_Bound'], (0, maxlim), (0, 10))
                    maxBound = np.interp(row['Max_Bound'], (0, maxlim), (0, 10))

                    reg = Rectangle((counter, minBound), width, maxBound-minBound, color = "red")
                    ax.add_patch(reg)
                    if count % 2 == 0:
                        ax.annotate("%s bp(%s)" % (formattedLength, row["No_Markers"]), 
                                    (xcenter,(maxBound+minBound)/2),  ha='right', 
                                    xytext=(xcenter+textPad, (maxBound+minBound)/2),
                                     arrowprops = {"arrowstyle" : '-'})
                    else:
                        ax.annotate("%s bp(%s)" % (formattedLength, row["No_Markers"]), 
                                    (xcenter,(maxBound+minBound)/2),  ha='left', 
                                    xytext=(xcenter-textPad, (maxBound+minBound)/2),
                                    arrowprops = {"arrowstyle" : '-'})
                    count += 1
            counter += pad
        ax.set_xticks(tickPos)        
        ax.set_xticklabels(["$%s$"%x for x in self.chromosomeSizes.keys()])
        ax.set_xlim(0,counter+margin)
        ax.set_ylim(-1,11)
            

def getExpectedGenotype(p1,p2):
    expectedGenotype = defaultdict(defaultdict)
    for chromosome in p1.keys():
        for pos in p1[chromosome]:
            expectedMarker = p1[chromosome][pos].union(p2[chromosome][pos])
            expectedGenotype[chromosome][pos] = expectedMarker

    return expectedGenotype

def getInformativeMarkers(p1,p2,child):
    informativeMarkers = rec_dd()
    for ch in child.keys():
        for pos in child[ch].keys():
            ip1 = p1[ch][pos]
            ip2 = p2[ch][pos]
            ichild = child[ch][pos]
            if ichild != ip1 or ichild != ip2:
                informativeMarkers['p1'][ch][pos] = ip1
                informativeMarkers['p2'][ch][pos] = ip2
                informativeMarkers['child'][ch][pos] = ichild
    return informativeMarkers


def rec_dd():
    return defaultdict(rec_dd)

def getMaxDicValue(sizes):
    maxi = 0
    for i in sizes.keys():
        if sizes[i] > maxi:
            maxi = sizes[i]
    return maxi



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the unknown origin regions of a child genotype based in the two parents genotype.')
    parser.add_argument("file", help="This dataset must be a csv delimitted by tabs")
    parser.add_argument("--p1", help="Name in the dataset of the parent 1")
    parser.add_argument("--p2", help="Name in the dataset of the parent 1")
    parser.add_argument("--child", help="Name in the dataset of the parent 1")
    parser.add_argument("--p11", help="Name in the dataset of the parent 1")
    parser.add_argument("--p12", help="Name in the dataset of the parent 1")
    parser.add_argument("--p21", help="Name in the dataset of the parent 2")
    parser.add_argument("--p22", help="Name in the dataset of the child")
    parser.add_argument("--threshold", type=float, default=2.0, help="The linkage distance threshold above which, clusters will not be merged")
    parser.add_argument("--outpath", help="Set the directory where the list of possible unknown regions going to storage", default = "./")
    parser.add_argument("--position", help="Method for especify the position of each marker", 
                            default="PRED"
                            ,choices = ["POS", "PRED"])

    


    args = parser.parse_args()
    df = pd.read_csv(args.file, sep='\t')   
    d = PedigreeRectificator(df, **args.__dict__)




