"""
Use this to build specific case files. Running this file will create 'case 7'
"""

# import statements
import os
import time
import h5py
import datetime
import warnings
import numpy as np
from copy import deepcopy
from scipy import spatial
from tagspace.clusters.makeclusters import makeclusters
from tagspace.wrappers.clusterfns import RandomAssign
from tagspace.wrappers.genfns import normalgeneration,choosestruct
from tagspace.data.spectra import psmspectra
from tagspace.data import gettimestr
from sklearn.cluster import DBSCAN, MiniBatchKMeans,KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from clustering_stats import *

# THEORETICAL INTRA CLUSTER SPREAD AT SNR=100                                                                                             
# From Ting et al 2016 arxiv:1602.06947
ch_cls = 5e-3                                                                                                 
nh_cls = 0.01
oh_cls = 0.01
nah_cls = 0.038
mgh_cls = 7.6e-3
alh_cls = 0.02
sih_cls = 8.2e-3
sh_cls = 0.024
kh_cls = 0.044
cah_cls = 0.016
tih_cls = 0.018
vh_cls = 0.06
mnh_cls = 0.013
nih_cls = 0.01
feh_cls = 4.3e-4
c12c13_cls = 0

tingspr = np.array([ch_cls,nh_cls,oh_cls,nah_cls,mgh_cls,alh_cls,sih_cls,sh_cls,kh_cls,
                    cah_cls,tih_cls,vh_cls,mnh_cls,nih_cls,feh_cls])

# INTRA CLUSTER SPREAD                                                                                                                    
# From 'global uncertainties' in Table 6 of Holtzmann et al 2015                                                                         
ch_cls = 3.5e-2
nh_cls = 0.067
oh_cls = 0.050
nah_cls = 0.064
mgh_cls = 0.053
alh_cls = 0.067
sih_cls = 0.077
sh_cls = 0.063
kh_cls = 0.065
cah_cls = 0.059
tih_cls = 0.072
vh_cls = 0.088
mnh_cls = 0.061
feh_cls = 0.053
nih_cls = 0.060
c12c13_cls = 0

spreads = np.array([ch_cls,nh_cls,oh_cls,nah_cls,mgh_cls,alh_cls,sih_cls,sh_cls,kh_cls,
                    cah_cls,tih_cls,vh_cls,mnh_cls,nih_cls,feh_cls])

ch_cls = 0.033
nh_cls = 0.031
oh_cls = 0.02
nah_cls = 0.049
mgh_cls = 0.017
alh_cls = 0.033
sih_cls = 0.018
sh_cls = 0.040
kh_cls = 0.021
cah_cls = 0.015
tih_cls = 0.019
vh_cls = 0.039
mnh_cls = 0.022
nih_cls = 0.017
feh_cls = 0.015

leungspr = np.array([ch_cls,nh_cls,oh_cls,nah_cls,mgh_cls,alh_cls,sih_cls,sh_cls,kh_cls,
                    cah_cls,tih_cls,vh_cls,mnh_cls,nih_cls,feh_cls])

elem = ['C','N','O','Na','Mg','Al','Si','S','K','Ca','Ti','V','Fe','Ni']
combelem = ['Na','Mg','Al','Si','S','K','Ca','Ti','V','Ni']

eigdata = np.load('eig20_minSNR50_corrNone_meanMed.pkl_data.npz')
eigvecs = eigdata['eigvec']
eigvals = eigdata['eigval']


tophats = np.load('tophat_elems.npy')
windows = np.load('window_elems.npy')
normeps = True

class fakeclusterclass(object):
    def __init__(self):
        self.timestamp = gettimestr()

def abuninds(elems,combelem):
    inds = []
    for e,elem in enumerate(elems):
        if elem in combelem:
            inds.append(e)
    return np.array(inds)

def combine_windows(windows = tophats,combelem=elem,func=np.ma.mean):
    """
    Combine windows from various elements into a single spectrum
    for dot product with spectrum.

    windows:    set of windows to use (tophats or actual windows)
    combelem:   list of elements to combine
    func:       function to use when combining the windows

    Returns spectrum   
    """
    windows = windows[abuninds(elem,combelem)]
    mask = windows == 0
    windows = np.ma.masked_array(windows,mask=mask)
    if func == np.ma.mean or func == np.ma.any:
        combspec = func(windows,axis=0)
    elif func == np.ma.max:
        combspec = func(windows.data,axis=0)
    return combspec.data


def create_indeps(mem,degree=2,full=np.array([]),cross=np.array([])):
    """
    Create array of independent variables, for a degree 1 or degree 2 
    polynomial excluding some terms.

    mem:        number of members per cluster
    degree:     degree of the polynomial
    full:       list of arrays that get full terms
    cross:      list of arrays that only get cross terms

    """
    polynomial = PolynomialFeatures(degree=degree)
    for i,indep in enumerate(full):
        full[i] = indep-np.median(indep)
    for i,indep in enumerate(cross):
        cross[i] = indep-np.median(indep)
    if degree==1:
        indeps = polynomial.fit_transform(full.T)
    elif degree==2:
        indeps = np.ones((mem,3*len(full)+len(full)*len(cross))).T
        pos = 0
        fullquad = polynomial.fit_transform(full.T)
        indeps[pos:pos+3*len(full)]=fullquad.T
        pos+=3*len(full)
        for findep in full:
            for cindep in cross:
                indeps[pos] = findep*cindep
                pos+=1
    return indeps.T
        

# run parameters
nstars = 1e3 # number of stars
sample='allStar_chemscrub.npy' # APOGEE sample to draw from
abundancefac = 0 # scaling factor for abundance noise
specfac = 0 # scaling factor for spectra noise
centerfac = 1
suff = 'H' # element denominator
metric = 'precomputed' # metric for distances
fullfitkeys = ['TEFF','LOGG'] # keys for the full fit
fullfitatms = []
crossfitkeys = []
crossfitatms = [6,7,8,11,12,13,14,16,19,20,22,23,25,26,28] # atomic numbers of cross terms
spreadchoice = spreads # choose which abudance spreads to employ

# DBSCAN parameters
min_samples = np.array([2,3])
samples = len(min_samples)
lg = np.arange(-3,1)
eps = [i*10.**lg for i in [1,5]]
eps = np.concatenate((eps[0],eps[1]))
eps.sort()
min_samples = np.tile(min_samples,len(eps))
eps = np.repeat(eps,samples)


class caserun(object):

    def __init__(self):
        self.timestamp = gettimestr()
        return None

    def makedata(self,nstars=nstars,clsind=2.1,volume=100,sample=sample,abundancefac=abundancefac,
                 spreadchoice=spreadchoice,specfac=specfac,centerfac=centerfac,
                 centerspr=spreads,genfn=choosestruct,
                 fullfitkeys=fullfitkeys,fullfitatms=fullfitatms,
                 crossfitkeys=crossfitkeys,crossfitatms=crossfitatms,
                 phvary=True,fitspec=True,case='7',add=False,usecenters=True):
        self.case = case
        start = time.time()
        self.create_clusters(nstars,clsind,volume,sample,genfn,centerfac,centerspr)
        end = time.time()
        print('Made clusters in {0} seconds'.format(end-start))
        start = time.time()
        self.create_stars(abundancefac,spreadchoice,specfac,phvary=phvary)
        end = time.time()
        print('Made stars in {0} seconds'.format(end-start))
        if add:
            self.insert_known_clusters(abundancefac,spreadchoice,usecenters=usecenters)
        if fitspec:
            start = time.time()
            self.fit_stars(fullfitkeys,fullfitatms,crossfitkeys,crossfitatms)
            end = time.time()
            print('Fit stars in {0} seconds'.format(end-start))
        self.plotfile()

    def importdata(self,fname):
        self.plot = h5py.File(fname)
        
    def create_clusters(self,nstars,clsind,volume,sample,genfn,centerfac,centerspr):
        self.nstars = nstars
        self.clsind = clsind
        self.sample = sample
        self.volume = volume
        # generate number of stars in a each cluster according to the CMF
        #os.system('python3.6 nstars.py -n {0} -a {1} -v {2}'.format(self.nstars,self.clsind,self.volume))

        # read in cluster info
        starfile = 'stararray.txt'
        scale = 1.0
        f = open(starfile)
        header = f.readline()[2:-1]
        f.close()
        self.numm = np.loadtxt(starfile).astype(int)
        self.numc = len(self.numm)
        self.mem = np.sum(self.numm)

        # designate true labels
        self.labels = np.arange(len(self.numm))
        self.labels_true = np.repeat(self.labels,self.numm,axis=0)

        self.propkeys=['C_{0}'.format(suff),
                       'N_{0}'.format(suff),
                       'O_{0}'.format(suff),
                       'NA_{0}'.format(suff),
                       'MG_{0}'.format(suff),
                       'AL_{0}'.format(suff),
                       'SI_{0}'.format(suff),
                       'S_{0}'.format(suff),
                       'K_{0}'.format(suff),
                       'CA_{0}'.format(suff),
                       'TI_{0}'.format(suff),
                       'V_{0}'.format(suff),
                       'MN_{0}'.format(suff),
                       'FE_H',
                       'NI_{0}'.format(suff)]

        # begin cluster generation by creating centers
        if genfn.__name__=='choosestruct':
            self.clusters = makeclusters(genfn=choosestruct,instances=1,
                                         numcluster=self.numc,maxcores=1,
                                         sample=self.sample,
                                         elems=np.array([6,7,8,11,12,13,
                                                         14,16,19,20,22,
                                                         23,25,26,28]),
                                         propkeys=self.propkeys)

        if genfn.__name__=='normalgeneration':
            self.clusters = makeclusters(genfn=normalgeneration,instances=1,
                                         numcluster=self.numc,maxcores=1,
                                         elems=np.array([6,7,8,11,12,13,
                                                         14,16,19,20,22,
                                                         23,25,26,28]),
                                         centers=np.zeros(15),
                                         stds=np.ones(15)*centerspr)
#        self.datafile = h5py.File(self.clusters.synfilename,'r+')
#        self.centers = self.datafile['center_abundances_'+self.clusters.timestamps[0].decode('UTF-8')]
        self.centers = self.clusters.centerdata[0]
        self.elems = np.array(self.propkeys)
        self.atmnums = np.array([6,7,8,11,12,13,14,16,19,20,22,23,25,26,28])
        self.elemnames = np.zeros(self.elems.shape,dtype='S2')
        self.centers[:]*=centerfac

        # Save true labels
        #dsetname = 'normalgeneration/labels_true_{0}'.format(self.clusters.timestamps[0].decode('UTF-8'))
        #self.datafile[dsetname] = self.labels_true

    def gen_abundances(self,abundancefac,spreadchoice,update=False):
        
        # Create abundances
        if not update:
            self.abundances = normalgeneration(num=self.mem,numprop=15,
                                               centers=np.repeat(self.centers[:],
                                                                 self.numm,
                                                                 axis=0),
                                               stds = abundancefac*spreadchoice)
        elif update:
            self.abundances[:self.mem] = normalgeneration(num=self.mem,numprop=15,
                                                          centers=np.repeat(self.centers[:],
                                                                            self.numm,
                                                                            axis=0),
                                                          stds = abundancefac*spreadchoice)
        

    def create_stars(self,abundancefac,spreadchoice,specfac,phvary=True):

        self.gen_abundances(abundancefac,spreadchoice)
        # Load in APOGEE data to generate stars with
        apodat = np.load(self.sample)
        # Figure out which APOGEE stars you want
        if self.mem > apodat.shape:
            warnings.warn('Every star in the input sample will be used to generate photospheres')
        if phvary:
            inds = np.random.randint(0,high=len(apodat),size=self.mem)
            d = apodat[inds]
            teffs = d['TEFF']
            loggs = d['LOGG']
        elif not phvary:
            teffs = [4900]*self.mem
            loggs = [2.7]*self.mem
        self.photosphere = np.array(list(zip(teffs,loggs)),
                                    dtype=[('TEFF','float'),('LOGG','float')])



        # Generate spectra
        self.specinfo = psmspectra(self.mem,self.photosphere,self.clusters.elems)
        self.specinfo.from_center_abundances(self.centers,self.numm,8)

        # Add noise to spectra
        self.specinfo.addnoise(normalgeneration,num=np.sum(self.numm),
                               numprop=7214,
                               centers = np.zeros(self.specinfo.spectra.shape),
                               stds = specfac*np.ones(self.specinfo.spectra.shape))

    def insert_known_clusters(self,abundancefac,spreadchoice,
                              clusterfname = 'occam_chemscrub_dr14.npy',
                              specfname='occam_chemscrub_dr14_interpspec.npy',
                              usecenters=True):
        prop = np.load(clusterfname)
        abundances = np.zeros((len(prop),len(self.propkeys)))
        for col,key in enumerate(self.propkeys):
            abundances[:,col] = prop[key]
        clusters = np.unique(prop['CLUSTER'])
        sizes = np.zeros(clusters.shape,dtype=int)
        start = self.labels[-1]+1
        labels = np.zeros(len(prop))
        centers = np.zeros((len(clusters),len(self.propkeys)))
        for c,cluster in enumerate(clusters):
            match = prop['CLUSTER']==cluster
            sizes[c] = np.sum(match)
            labels[match] = c+start
            centers[c] = np.median(abundances[match],axis=0)
        self.labels = np.arange(len(self.numm))
        self.labels_true = np.concatenate((self.labels_true,labels))
        self.mem += len(prop)
        self.numc += len(sizes)
        self.numm = np.concatenate((self.numm,sizes))
        #attrs =self.centers.attrs
        cens = np.concatenate((self.centers[:],centers))
        # Not as circular as it looks
        #self.datafile['center_abundances_wOCs_'+self.clusters.timestamps[0].decode('UTF-8')] = cens
        #self.centers = self.datafile['center_abundances_wOCs_'+self.clusters.timestamps[0].decode('UTF-8')]
        self.centers = cens
        # Hacky solution to put attributes back in for later fitting
        #for key in list(attrs.keys()):
        #    self.centers.attrs[key] = attrs[key]
        if usecenters:
            abudances = np.repeat(centers,sizes,axis=0)
        self.abundances = np.concatenate((self.abundances,abundances))
        if specfname:
            specs = np.load(specfname)
            photosphere = np.array(list(zip(prop['TEFF'],prop['LOGG'])),
                                    dtype=[('TEFF','float'),('LOGG','float')])
            self.specinfo.spectra = np.concatenate((self.specinfo.spectra,specs))
            self.photosphere = np.concatenate((self.photosphere,photosphere))
            


    def fit_stars(self,fullfitkeys,fullfitatms,crossfitkeys,crossfitatms):

        fulllist = []
        for i in fullfitkeys:
            fulllist.append(self.photosphere[i])

        for i in fullfitatms:
            elemcol = np.where(self.atmnums==i)[0]
            elem = np.repeat(self.centers[:,elemcol],self.numm)
            fulllist.append(elem)

        crosslist = []

        for i in crossfitkeys:
            crosslist.append(self.photosphere[i])

        for i in crossfitatms:
            elemcol = np.where(self.atmnums==i)[0]
            elem = np.repeat(self.centers[:,elemcol],self.numm)
            crosslist.append(elem)

        indeps = create_indeps(self.mem,full=np.array(fulllist),  cross=np.array(crosslist))

        # Fit out photospheric parameters
        coeff_terms = np.dot(np.dot(np.linalg.inv(np.dot(indeps.T,indeps)),indeps.T),self.specinfo.spectra)
        residual = self.specinfo.spectra - np.dot(indeps,coeff_terms)
        self.specinfo.spectra = residual

        # Save spectra
        #dsetname = 'normalgeneration/member_spectra_{0}'.format(self.clusters.timestamps[0].decode('UTF-8'))
        #self.datafile[dsetname] = self.specinfo.spectra

    def plotfile(self):
        self.pfname = 'case{0}_{1}.hdf5'.format(self.case,self.timestamp)
        self.plot = h5py.File(self.pfname,'w')
        self.plot['labels_true'] = self.labels_true
        self.plot['labels_true'].attrs['description'] = 'Integer for each star indicating its cluster membership'
        tcount,tlabs = membercount(self.labels_true)
        self.plot['true_size'] = tcount
        self.plot['true_size'].attrs['description'] = 'Size of each integer-labelled cluster'
        self.plot['centers'] = self.centers
        self.plot['centers'].attrs['description'] = 'Central abundances used to generate each integer-labelled cluster'

    def reduction(self,reduct=PCA,**kwargs):
        red = reduct(**kwargs)
        self.projectspec = red.fit_transform(self.specinfo.spectra)


    def projspec(self,arr,eigvals=None):
        if isinstance(arr,list):
            arr = np.array(arr)
        if isinstance(arr,np.ndarray):
            if len(arr.shape) == 1:
                arr = np.tile(arr,(self.mem,1))
                self.projectspec = arr*self.specinfo.spectra
            elif len(arr.shape) == 2:
                self.projspec = np.zeros(self.specinfo.spectra.shape)
                for a,r in arr:
                    if isinstance(eigvals,(list,np.ndarray)):
                        e = eigvals[a]
                    else:
                        e = 1
                    vec = np.tile(r,(self.mem,1))
                    self.projectspec += e*vec*self.specinfo.spectra

    def saveraw(self,arr,name,desc):
        self.plot[name] = arr
        self.plot[name].attrs['description'] = desc

    def calcdistances(self,arr,numuse='all'):
        if isinstance(numuse,(int,float)):
            numuse = int(numuse)
            inds = np.arange(len(arr))
            useinds = np.random.choice(np.arange(len(arr)),size=numuse,replace=False)
            self.distances = euclidean_distances(arr[useinds],arr[useinds])
        else:       
            self.distances = euclidean_distances(arr,arr)

    def cluster(self,arr,name,eps,min_samples,metric='precomputed',normeps=False,n_jobs=1,partname=None,savelabels=True):
        if name not in list(self.plot.keys()):
            self.plot[name] = arr

            self.plot.attrs['{0}_min'.format(name)] = min_samples
            self.plot.attrs['{0}_eps'.format(name)] = eps
    
        print(arr.shape)
        if not hasattr(self,'partname'):
            self.partname='nopart'

        if metric == 'precomputed':
            if not hasattr(self,'distances'): # FAILS ON RUNS WITH DIFFERENT DATATYPES, NEED NAME CHECK
                if arr.shape[0] > int(5.5e4):
                    metric='euclidean'
                    self.calcdistances(arr,numuse=1e4)
                elif arr.shape[0] <= int(5.5e4):
                    self.calcdistances(arr)
            if not hasattr(self,'typ'):        
                self.typ = np.median(self.distances)
        # Intialize predicted labels
        #d = distance_metrics(arr):
        labels_pred = -np.ones((len(eps),arr.shape[0]))
        for i in range(len(eps)):
            start = time.time()
            if metric =='precomputed':
                if not normeps:
                    db = DBSCAN(min_samples=min_samples[i],
                                eps=eps[i],n_jobs=n_jobs,
                                metric='precomputed').fit(self.distances)
                elif normeps:
                    db = DBSCAN(min_samples=min_samples[i],
                                eps=eps[i]*self.typ,
                                metric='precomputed').fit(self.distances)
            elif metric!='precomputed':
                db = DBSCAN(min_samples=min_samples[i],
                            eps=eps[i],n_jobs=n_jobs,
                            metric=metric).fit(arr)
            labels_pred[i] = db.labels_

        if savelabels:
            self.plot['{0}_labels_pred'.format(name)] = labels_pred
        end = time.time()
        print('Done DBSCAN {0} of {1} with eps {2} and min neighbours {3} on {4} - {5} seconds'.format(i+1,len(eps),eps[i],min_samples[i],name,np.round(end-start,2)))
        return labels_pred

    def kmeanspart(self,arr,num=2,radiscale=1):
        self.partname = 'kmeans'
        tree = spatial.cKDTree(arr)
        k = KMeans(n_clusters=num).fit(arr)
        means = k.cluster_centers_
        potential_overlap = tree.query_ball_point(means, radiscale*self.typ)
        sectiontags = []
        sections = []
        inds = np.arange(len(arr))
        for n in range(num):
            indn = inds[k.labels_==n]
            allinds = np.unique(np.concatenate((indn,potential_overlap[n])))
            sectiontags.append(allinds)
            sections.append(arr[allinds])
        return sections,sectiontags
        
    def custompart(self,arr,num=2,size=int(3e4)):
        self.partname = 'custom'
        sections = []
        sectiontags = []
        starters = []
        tree = spatial.cKDTree(arr)
        np.random.seed(1)
        ind = np.random.choice(np.arange(len(arr)),replace=False)
        k = arr.shape[0]
        distances,indices = tree.query(arr[ind],k=k)
        sortinds = np.argsort(indices)
        distances = distances[sortinds]
        indices = indices[sortinds]
        maxdist =  np.max(distances[np.invert(np.isinf(distances))])
        maxdistind = np.where(np.fabs(distances-maxdist<1e-12))[0][0]
        d,sectioninds = tree.query(arr[maxdistind],k=size)
        starters.append(maxdistind)
        sectioninds.sort()
        sectiontags.append(sectioninds)
        sections.append(arr[sectioninds])
        reducedarr = np.delete(arr,sectioninds,axis=0)
        reducedinds = np.delete(np.arange(arr.shape[0]),sectioninds)
        # labs = self.cluster(sections[0],name+'part1',eps,min_samples,metric=metric,normeps=normeps,n_jobs=n_jobs)
        # for e in range(len(eps)):
        #     self.labels_pred[0][e][sectioninds] = labs[e]

        for i in np.arange(1,num):
            # Repeat for maximally separated PC30
            reducedtree = spatial.cKDTree(reducedarr)
            start = time.time()
            distances,indices = reducedtree.query(arr[maxdistind],k=reducedarr.shape[0])
            end = time.time()
            print('found neighbours in {0} s'.format(end-start))
            sortinds = np.argsort(indices)
            distances = distances[sortinds]
            indices = indices[sortinds]
            # Find max non infinite distance
            maxdist =  np.max(distances[np.invert(np.isinf(distances))])
            maxdistind = np.where(np.fabs(distances-maxdist<1e-12))[0][0]
            # NEED TO TRANSLATE IND
            d,newsectioninds = tree.query(arr[reducedinds[maxdistind]],k=size)
            starters.append(reducedinds[maxdistind])
            newsectioninds.sort()
            sectiontags.append(newsectioninds)
            sections.append(arr[newsectioninds])
            sectioninds = np.concatenate((sectioninds,newsectioninds))
            sectioninds.sort()
            reducedarr = np.delete(arr,sectioninds,axis=0)
            reducedinds = np.delete(np.arange(arr.shape[0]),sectioninds)
            # labs = self.cluster(sections[i],name+'part{0}'.format(i+1),eps,min_samples,metric=metric,normeps=normeps,n_jobs=n_jobs)
            # for e in range(len(eps)):
            #     self.labels_pred[i][e][newsectioninds] = labs[e]
        print('{0} still unassigned'.format(len(reducedarr)))

        print(sections[0].shape)
        print(sectiontags[0].shape)
        parttrees = {}
        for n in range(num):
            parttrees[n] = spatial.cKDTree(sections[n])
        separations = []
        for s,star in enumerate(reducedarr):
            partdists = []
            for n in range(num):
                parttree = parttrees[n]
                distances,indices = parttree.query(star,k=2)
                partdists.append(distances[1])
            separations.append(np.min(np.array(partdists)))

        checks = 100
        associates = 50
        sepsort = np.argsort(np.array(separations))
        for s,star in enumerate(reducedarr[sepsort]):
            distances,indices = tree.query(star,k=checks)
            locs = []
            for i in indices:
                for sectag in range(len(sectiontags)):
                    if i in sectiontags[sectag]:
                        locs.append(sectag)
            if locs == []:
                #nolabel.append(star)
                print('no neighbours!')
            locs = np.array(locs)
            possible = np.unique(locs[:associates])
            for p in possible:
                #print(sections[p].shape,star,star.shape,np.array(star).shape)
                sectiontags[p] = np.append(sectiontags[p],np.array([indices[0]]))
                sections[p] = np.append(sections[p],np.array([star]),axis=0)
        return sections,sectiontags

    def findoverlap(self,lenarr,sectiontags):
        #find overlap and connected stars
        overlap = []
        for i in range(lenarr):
            matchx = []
            matchy = []
            for n in range(len(sectiontags)):
                match = np.where(sectiontags[n]==i)
                if len(match[0])>=1:
                    matchx.append(n)
                    matchy.append(match)
            matchlist = (np.array(matchx),np.array(matchy))
            if len(matchlist[0])>1:
                overlap.append(i)
        return overlap
                    
    def findconnected(self,sectiontags,overlap,labels_pred):
        fullconnected = []
        for e in range(len(labels_pred[0])):
            connected = np.array([],dtype=int)
            for n in range(len(sectiontags)):
                connect = {}
                for o in range(len(overlap)):
                    match = np.where(sectiontags[n]==overlap[o])
                    label = labels_pred[n][e][match][0]
                    if label not in list(connect.keys()):
                        connect[label] = sectiontags[n][labels_pred[n][e]==label]
                inds = np.array([i for sublist in list(connect.values()) for i in sublist])
                connected = np.append(connected,inds)
            fullconnected.append(np.unique(connected))
        return fullconnected


    def partition(self,num,arr,name,eps,min_samples,partitionfns=[],partitionkwargs = [],metric='precomputed',neighbours=20,normeps=False,n_jobs=1,seed=1,total=True):

        if arr.shape[0] > int(5.5e4):
            metric='euclidean'
            self.calcdistances(arr,numuse=1e4)
        elif arr.shape[0] <= int(5.5e4):
            self.calcdistances(arr)
            self.typ = np.median(self.distances)

        if total:
            self.total_labels_pred = self.cluster(arr,name,eps,min_samples,metric=metric,normeps=normeps,n_jobs=n_jobs)
            self.plot['{0}_total_labels_pred'.format(name)]= self.total_labels_pred

        for p,partitionfn in enumerate(partitionfns):
            sections,sectiontags = partitionfn(arr,**partitionkwargs[p])
            overlap = self.findoverlap(len(arr),sectiontags)

            labels_pred = []
            self.part_labels_pred = -np.ones(self.total_labels_pred.shape)

            for n in range(num):
                print(np.array(sections[n]).shape)
                nooverlap = [i for i in range(len(sectiontags[n])) if i not in overlap]
                self.calcdistances(sections[n])
                labs = self.cluster(np.array(sections[n]),name+'_'+self.partname+'_part{0}'.format(n+1),eps,min_samples,metric=metric,normeps=normeps,n_jobs=n_jobs)
                labels_pred.append(labs)
                for e in range(len(labs)):
                    labscale = np.max( self.part_labels_pred[e])+1
                    self.part_labels_pred[e][sectiontags[n][nooverlap]] = labs[e][nooverlap] + labscale
            connected = self.findconnected(sectiontags,overlap,labels_pred)

            for e in range(len(labels_pred[0])):
                self.calcdistances(arr[connected[e]])
                labs = self.cluster(arr[connected[e]],name+'_'+self.partname+'_overlap',[eps[e]],[min_samples[e]],metric=metric,normeps=normeps,n_jobs=n_jobs,savelabels=False)
                labscale = np.max( self.part_labels_pred[e])+1
                self.part_labels_pred[e][connected[e]]=labs[0]+labscale

            self.plot['{0}_{1}_connect_labels_pred'.format(name,self.partname)]= self.part_labels_pred

    def clustering(self,arr,name,eps,min_samples,metric='precomputed',neighbours = 20,normeps=False,n_jobs=1,clustersknown=True):

        if name not in list(self.plot.keys()):
            self.plot[name] = arr

            self.plot.attrs['{0}_min'.format(name)] = min_samples
            self.plot.attrs['{0}_eps'.format(name)] = eps

        # Generate distances if using precomputed metric
        if metric=='precomputed':
            distances = euclidean_distances(arr,arr)
            typ = np.median(distances)
            

        # Intialize predicted labels
        d = distance_metrics(arr)
        labels_pred = -np.ones((len(eps),arr.shape[0]))
        cbn = np.zeros((len(eps),arr.shape[0]))
        for i in range(len(eps)):
            start = time.time()
            if metric =='precomputed':
                if not normeps:
                    db = DBSCAN(min_samples=min_samples[i],
                                eps=eps[i],n_jobs=n_jobs,
                                metric='precomputed').fit(distances)
                elif normeps:
                    db = DBSCAN(min_samples=min_samples[i],
                                eps=eps[i]*typ,
                                metric='precomputed').fit(distances)
            elif metric!='precomputed':
                db = DBSCAN(min_samples=min_samples[i],
                            eps=eps[i],n_jobs=n_jobs,
                            metric=metric).fit(arr)
        
            labels_pred[i] = db.labels_ 
        self.plot['{0}_labels_pred'] = labels_pred

    # def cluster
    #         efficiency, completeness, plabs, matchtlabs = efficiency_completeness(db.labels_,self.labels_true,minmembers=1)
    #         if len(plabs) > 5:
    #             k = neighbours
    #             if len(plabs) < neighbours and len(plabs) > 1:
    #                 k = len(plabs)-1
    #             elif len(plabs) == 1:
    #                 k = 1
    #             self.plot['{0}_match_tlabs_eps{1}_min{2}'.format(name,eps[i],min_samples[i])] = matchtlabs
    #             self.plot['{0}_found_sil_eps{1}_min{2}_neigh{3}'.format(name,eps[i],min_samples[i],neighbours)] = d.silhouette(db.labels_,k=k)[0]
    #             self.plot['{0}_eff_eps{1}_min{2}'.format(name,eps[i],min_samples[i])] = efficiency
    #             self.plot['{0}_com_eps{1}_min{2}'.format(name,eps[i],min_samples[i])] = completeness
    #             self.plot['{0}_found_size_eps{1}_min{2}'.format(name,eps[i],min_samples[i])] = pcount
    #         elif len(plabs) <= 5:
    #             self.plot['{0}_match_tlabs_eps{1}_min{2}'.format(name,eps[i],min_samples[i])] = np.array([])
    #             self.plot['{0}_found_sil_eps{1}_min{2}_neigh{3}'.format(name,eps[i],min_samples[i],neighbours)] = np.array([])
    #             self.plot['{0}_eff_eps{1}_min{2}'.format(name,eps[i],min_samples[i])] = np.array([])
    #             self.plot['{0}_com_eps{1}_min{2}'.format(name,eps[i],min_samples[i])] = np.array([])
    #             self.plot['{0}_found_size_eps{1}_min{2}'.format(name,eps[i],min_samples[i])] = np.array([])
    #         core = db.core_sample_indices_
    #         cbn[i][core] = 1
                   
    #         noise = np.where(db.labels_==-1)
    #         cbn[i][noise] = -1
    #         noise = np.where(db.labels_==-1)
    #         end = time.time()
    #         print('Done DBSCAN {0} of {1} with eps {2} and min neighbours {3} on {4} - {5} seconds'.format(i+1,len(eps),eps[i],min_samples[i],name,np.round(end-start,2)))
    #         print('I found {0} out of {1} clusters'.format(len(plabs),self.numc)) 
        
    #     self.plot['{0}_cbn'.format(name)] = cbn

    def finish(self):
        #self.datafile.close()
        self.plot.close()
        print('I saved everything in {0}'.format(self.pfname))

if __name__=='__main__':
    case7 = caserun()
    start = time.time()
    """
    case7.clustering(case7.specinfo.spectra,'spec',eps,min_samples,metric='precomputed',
                    neighbours = 20,normeps=normeps)
    case7.clustering(case7.abundances,'abun',eps,min_samples,metric='precomputed',
                    neighbours = 20,normeps=normeps)
    toph = combine_windows(windows = tophats,combelem=combelem,func=np.ma.any)
    case7.projspec(toph)
    case7.clustering(case7.projectspec,'toph',eps,min_samples,metric='precomputed',
                    neighbours = 20,normeps=normeps)
    case7.reduction(reduct = PCA, n_components=10)
    case7.clustering(case7.projectspec,'prin',eps,min_samples,metric='precomputed',
                    neighbours = 20,normeps=normeps)
    """   
    end = time.time()
#    case7.finish()
    print('Finished desired clustering in {0} seconds'.format(end-start))
