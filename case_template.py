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
from scipy import spatial
from tagspace.clusters.makeclusters import makeclusters
from tagspace.wrappers.clusterfns import RandomAssign
from tagspace.wrappers.genfns import normalgeneration,choosestruct
from tagspace.data.spectra import psmspectra
from tagspace.data import gettimestr
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import PolynomialFeatures
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
ch_cls = 0.035
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
suff = 'H' # element denominator
metric = 'precomputed' # metric for distances
fullfitkeys = ['TEFF','LOGG'] # keys for the full fit
fullfitatms = []
crossfitkeys = []
crossfitatms = [6,7,8,11,12,13,14,16,19,20,22,23,25,26,28] # atomic numbers of cross terms
spreadchoice = spreads # choose which abudance spreads to employ

# DBSCAN parameters
smin_samples = np.array([2,3])
ssamples = len(smin_samples)
lg = np.arange(-3,1)
eps = [i*10.**lg for i in [1,5]]
eps = np.concatenate((eps[0],eps[1]))
eps.sort()
seps = eps
smin_samples = np.tile(smin_samples,len(seps))
amin_samples = np.array([2,3])
asamples = len(amin_samples)
aeps = eps
amin_samples = np.tile(amin_samples,len(aeps))
aeps = np.repeat(aeps,asamples)
seps = np.repeat(seps,ssamples)


class caserun(object):

    def __init__(self,nstars=nstars,sample=sample,abundancefac=abundancefac,
                 spreadchoice=spreadchoice,specfac=specfac,
                 fullfitkeys=fullfitkeys,fullfitatms=fullfitatms,
                 crossfitkeys=crossfitkeys,crossfitatms=crossfitatms,
                 seps=seps,smin_samples=smin_samples,
                 aeps=aeps,amin_samples=amin_samples,
                 metric='precomputed',neighbours = 20,phvary=True,
                 fitspec=True,case='7'):
        self.case = case
        start = time.time()
        self.create_clusters(nstars,sample)
        end = time.time()
        print('Made clusters in {0} seconds'.format(end-start))
        start = time.time()
        self.create_stars(abundancefac,spreadchoice,specfac,phvary=phvary)
        end = time.time()
        print('Made stars in {0} seconds'.format(end-start))
        if fitspec:
            start = time.time()
            self.fit_stars(fullfitkeys,fullfitatms,crossfitkeys,crossfitatms)
            end = time.time()
            print('Fit stars in {0} seconds'.format(end-start))
        self.plotfile()
        start = time.time()
        self.clustering(seps,aeps,smin_samples,amin_samples,metric='precomputed',neighbours = 20)
        end = time.time()
        print('Finished desired clustering in {0} seconds'.format(end-start))


    def create_clusters(self,nstars,sample):
        self.nstars = nstars
        self.sample = sample
        # generate number of stars in a each cluster according to the CMF
        os.system('python3 nstars.py -n {0}'.format(self.nstars))

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

        # begin cluster generation by creating centers
        self.clusters = makeclusters(genfn=choosestruct,instances=1,
                                     numcluster=self.numc,maxcores=1,
                                     sample=self.sample,
                                     elems=np.array([6,7,8,11,12,13,14,16,
                                                     19,20,22,23,25,26,28]),
                                     propkeys=['C_{0}'.format(suff),
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
                                               'FE_H','NI_{0}'.format(suff)])

        # Grab data file to save more stuff in it
        self.datafile = h5py.File(self.clusters.synfilename,'r+')
        self.centers = self.datafile['center_abundances_'+self.clusters.timestamps[0].decode('UTF-8')]
        # Save true labels
        dsetname = 'normalgeneration/labels_true_{0}'.format(self.clusters.timestamps[0].decode('UTF-8'))
        self.datafile[dsetname] = self.labels_true

    def create_stars(self,abundancefac,spreadchoice,specfac,phvary=True):

        # Create abundances
        self.abundances = normalgeneration(num=self.mem,numprop=15,
                                           centers=np.repeat(self.centers,
                                                             self.numm,
                                                             axis=0),
                                           stds = abundancefac*spreadchoice)

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

    def fit_stars(self,fullfitkeys,fullfitatms,crossfitkeys,crossfitatms):

        fulllist = []
        for i in fullfitkeys:
            fulllist.append(self.photosphere[i])

        for i in fullfitatms:
            elemcol = np.where(self.centers.attrs['atmnums']==i)[0]
            elem = np.repeat(self.centers[:,elemcol],self.numm)
            fulllist.append(elem)

        crosslist = []

        for i in crossfitkeys:
            crosslist.append(self.photosphere[i])

        for i in crossfitatms:
            elemcol = np.where(self.centers.attrs['atmnums']==i)[0]
            elem = np.repeat(self.centers[:,elemcol],self.numm)
            crosslist.append(elem)

        indeps = create_indeps(self.mem,full=np.array(fulllist),  cross=np.array(crosslist))

        # Fit out photospheric parameters
        coeff_terms = np.dot(np.dot(np.linalg.inv(np.dot(indeps.T,indeps)),indeps.T),self.specinfo.spectra)
        residual = self.specinfo.spectra - np.dot(indeps,coeff_terms)
        self.specinfo.spectra = residual

        # Save spectra
        dsetname = 'normalgeneration/member_spectra_{0}'.format(self.clusters.timestamps[0].decode('UTF-8'))
        self.datafile[dsetname] = self.specinfo.spectra
        self.datafile.close()

    def plotfile(self):
        self.pfname = 'case{0}_{1}.hdf5'.format(self.case,
                                                self.clusters.timestamps[0].decode('UTF-8'))
        self.plot = h5py.File(self.pfname,'w')
        self.plot['labels_true'] = self.labels_true
        tcount,tlabs = membercount(self.labels_true)
        self.plot['true_size'] = tcount

    def clustering(self,seps,aeps,smin_samples,amin_samples,metric='precomputed',neighbours = 20):

        self.plot.attrs['spec_min'] = smin_samples
        self.plot.attrs['spec_eps'] = seps
        self.plot.attrs['abun_min'] = amin_samples
        self.plot.attrs['abun_eps'] = aeps

        # Generate distances if using precomputed metric
        if metric=='precomputed':
            spec_distances = euclidean_distances(self.specinfo.spectra, 
                                                 self.specinfo.spectra)
            abun_distances = euclidean_distances(self.abundances,
                                                 self.abundances)

        # Intialize predicted labels
        # d = distance_metrics(self.specinfo.spectra)
        #self.plot['spec_true_sil_neigh{0}'.format(neighbours)] = d.silhouette(self.labels_true,k=neighbours)[0]
        spec_labels_pred = -np.ones((len(seps),self.mem))
        spec_cbn = np.zeros((len(seps),self.mem))
        for i in range(len(seps)):
            start = time.time()
            if metric =='precomputed':
                db = DBSCAN(min_samples=smin_samples[i],
                            eps=seps[i],
                            metric='precomputed').fit(spec_distances)
            elif metric!='precomputed':
                db = DBSCAN(min_samples=smin_samples[i],
                            eps=seps[i],
                            metric=metric).fit(self.specinfo.spectra)
            pcount,plabs = membercount(db.labels_)
            bad = np.where(plabs==-1)
            if len(bad[0])>0:
                plabs = np.delete(plabs,bad[0][0])
                pcount = np.delete(pcount,bad[0][0])
            efficiency, completeness, plabs, matchtlabs = efficiency_completeness(db.labels_,self.labels_true,minmembers=1)
            if len(plabs) > 5:
                k = neighbours
                if len(plabs) < neighbours and len(plabs) > 1:
                    k = len(plabs)-1
                elif len(plabs) == 1:
                    k = 1
                self.plot['spec_match_tlabs_eps{0}_min{1}'.format(seps[i],smin_samples[i])] = matchtlabs
                #self.plot['spec_found_sil_eps{0}_min{1}_neigh{2}'.format(seps[i],smin_samples[i],k)] = d.silhouette(spec_labels_pred[i],k=k)[0]
                self.plot['spec_eff_eps{0}_min{1}'.format(seps[i],smin_samples[i])] = efficiency
                self.plot['spec_com_eps{0}_min{1}'.format(seps[i],smin_samples[i])] = completeness
                self.plot['spec_found_size_eps{0}_min{1}'.format(seps[i],smin_samples[i])] = pcount
            elif len(plabs) <= 5:
                self.plot['spec_match_tlabs_eps{0}_min{1}'.format(seps[i],smin_samples[i])] = np.array([])
                #self.plot['spec_found_sil_eps{0}_min{1}_neigh{2}'.format(seps[i],smin_samples[i],k)] = np.array([])
                self.plot['spec_eff_eps{0}_min{1}'.format(seps[i],smin_samples[i])] = np.array([])
                self.plot['spec_com_eps{0}_min{1}'.format(seps[i],smin_samples[i])] = np.array([])
                self.plot['spec_found_size_eps{0}_min{1}'.format(seps[i],smin_samples[i])] = np.array([])
            core = db.core_sample_indices_
            spec_cbn[i][core] = 1
            spec_labels_pred[i] = db.labels_        
            noise = np.where(db.labels_==-1)
            spec_cbn[i][noise] = -1
            noise = np.where(db.labels_==-1)
            end = time.time()
            print('Done DBSCAN {0} of {1} with eps {2} and min neighbours {3} - {4} seconds'.format(i+1,len(seps),seps[i],smin_samples[i],np.round(end-start,2)))
            print('I found {0} out of {1} clusters'.format(len(plabs),self.numc)) 
        self.plot['spec_labels_pred'] = spec_labels_pred
        self.plot['spec_cbn'] = spec_cbn

        # Intialize predicted labels
#        d = distance_metrics(self.abundances)
        #self.plot['abun_true_sil_neigh{0}'.format(neighbours)] = d.silhouette(self.labels_true,k=neighbours)[0]
        abun_labels_pred = -np.ones((len(aeps),self.mem))
        abun_cbn = np.zeros((len(aeps),self.mem))
        for i in range(len(aeps)):
            start = time.time()
            if metric =='precomputed':
                db = DBSCAN(min_samples=amin_samples[i],
                            eps=aeps[i],
                            metric='precomputed').fit(abun_distances)
            elif metric!='precomputed':
                db = DBSCAN(min_samples=amin_samples[i],
                            eps=aeps[i],
                            metric=metric).fit(self.abundances)
            pcount,plabs = membercount(db.labels_)
            bad = np.where(plabs==-1)
            if len(bad[0])>0:
                plabs = np.delete(plabs,bad[0][0])
                pcount = np.delete(pcount,bad[0][0])
            efficiency, completeness, plabs, matchtlabs = efficiency_completeness(db.labels_,self.labels_true,minmembers=1)
            if len(plabs) > 5:
                k = neighbours
                if len(plabs) < neighbours and len(plabs) > 1:
                    k = len(plabs)-1
                elif len(plabs) == 1:
                    k = 1
                self.plot['abun_match_tlabs_eps{0}_min{1}'.format(aeps[i],amin_samples[i])] = matchtlabs
                #self.plot['abun_found_sil_eps{0}_min{1}_neigh{2}'.format(aeps[i],amin_samples[i],neighbours)] = d.silhouette(adun_labels_pred[i],k=k)[0]
                self.plot['abun_eff_eps{0}_min{1}'.format(aeps[i],amin_samples[i])] = efficiency
                self.plot['abun_com_eps{0}_min{1}'.format(aeps[i],amin_samples[i])] = completeness
                self.plot['abun_found_size_eps{0}_min{1}'.format(aeps[i],amin_samples[i])] = pcount
            elif len(plabs) <= 5:
                self.plot['abun_match_tlabs_eps{0}_min{1}'.format(aeps[i],amin_samples[i])] = np.array([])
                #self.plot['abun_found_sil_eps{0}_min{1}_neigh{2}'.format(aeps[i],amin_samples[i],neighbours)] = np.array([])
                self.plot['abun_eff_eps{0}_min{1}'.format(aeps[i],amin_samples[i])] = np.array([])
                self.plot['abun_com_eps{0}_min{1}'.format(aeps[i],amin_samples[i])] = np.array([])
                self.plot['abun_found_size_eps{0}_min{1}'.format(aeps[i],amin_samples[i])] = np.array([])
            core = db.core_sample_indices_
            abun_cbn[i][core] = 1
            abun_labels_pred[i] = db.labels_        
            noise = np.where(db.labels_==-1)
            abun_cbn[i][noise] = -1
            noise = np.where(db.labels_==-1)
            end = time.time()
            print('Done DBSCAN {0} of {1} with eps {2} and min neighbours {3} - {4} seconds'.format(i+1,len(aeps),seps[i],amin_samples[i],np.round(end-start,2)))
            print('I found {0} out of {1} clusters'.format(len(plabs),self.numc)) 
        self.plot['abun_labels_pred'] = abun_labels_pred
        self.plot['abun_cbn'] = abun_cbn
        self.plot.close()
        print('I saved everything in {0}'.format(self.pfname))

if __name__=='__main__':
    case7 = caserun()
