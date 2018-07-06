# import statements
import os
import time
import h5py
import datetime
import warnings
import numpy as np
from scipy import spatial
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
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

def create_indeps(mem,degree=2,full=np.array([]),cross=np.array([])):
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
nstars = 5e4 # number of stars
sample='allStar_chemscrub.npy' # APOGEE sample to draw from
abundancefac = 0 # scaling factor for abundance noise
specfac = 0 # scaling factor for spectra noise
suff = 'H' # element denominator
metric = 'precomputed' # metric for distances
fullfitkeys = ['TEFF','LOGG']
crossfitatms = [6,7,8,11,12,13,14,16,19,20,22,23,25,26,28]

# DBSCAN parameters
smin_samples = np.array([2,3])#,5,10])#,15,20,50])
ssamples = len(smin_samples)
seps = np.array([0.05,0.075,0.1])#,0.5,1.0])
smin_samples = np.tile(smin_samples,len(seps))
amin_samples = np.array([2,3])#,4,5,6])
asamples = len(amin_samples)
aeps = np.array([0.1,0.3,0.35])#,0.4,0.5])
amin_samples = np.tile(amin_samples,len(aeps))
aeps = np.repeat(aeps,asamples)
seps = np.repeat(seps,ssamples)

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

# choose which abudance spreads to employ
spreadchoice = spreads

# generate number of stars in a each cluster according to the CMF
os.system('python3 nstars.py -n {0}'.format(nstars))

# read in cluster info
starfile = 'stararray.txt'
scale = 1.0
f = open(starfile)
header = f.readline()[2:-1]
f.close()
numm = np.loadtxt(starfile).astype(int)
numc = len(numm)
mem = np.sum(numm)

# designate true labels
labels = np.arange(len(numm))
labels_true = np.repeat(labels,numm,axis=0)

# begin cluster generation by creating centers
clusters = makeclusters(genfn=choosestruct,instances=1,numcluster=numc,maxcores=1,
                elems=np.array([6,7,8,11,12,13,14,16,19,20,22,23,25,26,28]),
                        sample=sample,
                        propkeys=['C_{0}'.format(suff),'N_{0}'.format(suff),'O_{0}'.format(suff),'NA_{0}'.format(suff),
                                  'MG_{0}'.format(suff),'AL_{0}'.format(suff),'SI_{0}'.format(suff),'S_{0}'.format(suff),
                                  'K_{0}'.format(suff),'CA_{0}'.format(suff),'TI_{0}'.format(suff),'V_{0}'.format(suff),
                                  'MN_{0}'.format(suff),'FE_H','NI_{0}'.format(suff)])

# Grab data file to save more stuff in it
datafile = h5py.File(clusters.synfilename,'r+')
centers = datafile['center_abundances_'+clusters.timestamps[0].decode('UTF-8')]

# Create abundances
abundances = normalgeneration(num=mem,numprop=15,
                              centers=np.repeat(centers,numm,axis=0),
                              stds = abundancefac*spreadchoice)

# Save true labels
dsetname = 'normalgeneration/labels_true_{0}'.format(clusters.timestamps[0].decode('UTF-8'))
datafile[dsetname] = labels_true

# Load in APOGEE data to generate stars with
apodat = np.load(sample)
# Figure out which APOGEE stars you want
if mem > apodat.shape:
    warnings.warn('Every star in the input sample will be used to generate photospheres')
inds = np.random.randint(0,high=len(apodat),size=mem)
d = apodat[inds]
photosphere = np.array(list(zip(d['TEFF'],d['LOGG'])),dtype=[('TEFF','float'),('LOGG','float')])

# Generate spectra
specinfo = psmspectra(mem,photosphere,clusters.elems)
specinfo.from_center_abundances(centers,numm,8)

# Add noise to spectra
specinfo.addnoise(normalgeneration,num=np.sum(numm),numprop=7214,
                  centers = np.zeros(specinfo.spectra.shape), 
                  stds = specfac*np.ones(specinfo.spectra.shape))

teffs = photosphere['TEFF']
loggs = photosphere['LOGG']

# Extract chemistry columns
chcol = np.where(centers.attrs['atmnums']==6)[0]
chs = np.repeat(centers[:,chcol],numm)

nhcol = np.where(centers.attrs['atmnums']==7)[0]
nhs = np.repeat(centers[:,nhcol],numm)

ohcol = np.where(centers.attrs['atmnums']==8)[0]
ohs = np.repeat(centers[:,ohcol],numm)

nahcol = np.where(centers.attrs['atmnums']==11)[0]
nahs = np.repeat(centers[:,nahcol],numm)

mghcol = np.where(centers.attrs['atmnums']==12)[0]
mghs = np.repeat(centers[:,mghcol],numm)

alhcol = np.where(centers.attrs['atmnums']==13)[0]
alhs = np.repeat(centers[:,alhcol],numm)

sihcol = np.where(centers.attrs['atmnums']==14)[0]
sihs = np.repeat(centers[:,sihcol],numm)

shcol = np.where(centers.attrs['atmnums']==16)[0]
shs = np.repeat(centers[:,shcol],numm)

khcol = np.where(centers.attrs['atmnums']==19)[0] 
khs = np.repeat(centers[:,khcol],numm)

cahcol = np.where(centers.attrs['atmnums']==20)[0]
cahs = np.repeat(centers[:,cahcol],numm)

tihcol = np.where(centers.attrs['atmnums']==22)[0]
tihs = np.repeat(centers[:,tihcol],numm)

vhcol = np.where(centers.attrs['atmnums']==23)[0]
vhs = np.repeat(centers[:,vhcol],numm)

mnhcol = np.where(centers.attrs['atmnums']==25)[0]
mnhs = np.repeat(centers[:,mnhcol],numm)

fehcol = np.where(centers.attrs['atmnums']==26)[0]
fehs = np.repeat(centers[:,fehcol],numm)

nihcol = np.where(centers.attrs['atmnums']==28)[0]
nihs = np.repeat(centers[:,nihcol],numm)

indeps = create_indeps(mem,full=np.array([teffs,loggs]),  cross=np.array([chs,nhs,ohs,nahs,mghs,alhs,sihs,shs,khs,cahs,tihs,vhs,mnhs,fehs,nihs]))

# Fit out photospheric parameters
coeff_terms = np.dot(np.dot(np.linalg.inv(np.dot(indeps.T,indeps)),indeps.T),specinfo.spectra)
residual = specinfo.spectra - np.dot(indeps,coeff_terms)
specinfo.spectra = residual

# Save spectra
dsetname = 'normalgeneration/member_spectra_{0}'.format(clusters.timestamps[0].decode('UTF-8'))
datafile[dsetname] = specinfo.spectra

# Generate distances if using precomputed metric
if metric=='precomputed':
    spec_distances = euclidean_distances(specinfo.spectra, specinfo.spectra)
    abun_distances = euclidean_distances(abundances,abundances)

# Intialize predicted labels
spec_labels_pred = -np.ones((len(seps),mem))
cen_labels_pred = -np.ones((len(aeps),mem))
spec_cbn = np.zeros((len(seps),mem))
cen_cbn = np.zeros((len(aeps),mem))
for i in range(len(seps)):
    start = time.time()
    if metric =='precomputed':
        db = DBSCAN(min_samples=smin_samples[i],
                    eps=seps[i],metric='precomputed').fit(spec_distances)
    elif metric!='precomputed':
        db = DBSCAN(min_samples=smin_samples[i],
                    eps=seps[i],metric=metric).fit(specinfo.spectra)
    core = db.core_sample_indices_
    spec_cbn[i][core] = 1
    spec_labels_pred[i] = db.labels_
    noise = np.where(db.labels_==-1)
    spec_cbn[i][noise] = -1
    pcount,plabs = membercount(db.labels_)
    bad = np.where(plabs==-1)
    if len(bad[0])>0:
        plabs = np.delete(plabs,bad[0][0])
        pcount = np.delete(pcount,bad[0][0])
    noise = np.where(db.labels_==-1)
    end = time.time()
    print('Done DBSCAN {0} of {1} with eps {2} and min neighbours {3} - {4} seconds'.format(i+1,len(seps),seps[i],smin_samples[i],np.round(end-start,2)))
    print('I found {0} out of {1} clusters'.format(len(plabs),numc)) 
    
for i in range(len(aeps)):
    start = time.time()
    if metric=='precomputed':
        db = DBSCAN(min_samples=amin_samples[i],
                    eps=aeps[i],metric='precomputed').fit(abun_distances)
    elif metric!='precomputed':
        db = DBSCAN(min_samples=amin_samples[i],
                    eps=aeps[i],metric=metric).fit(abundances)
    core = db.core_sample_indices_
    cen_cbn[i][core] = 1
    cen_labels_pred[i] = db.labels_
    noise = np.where(db.labels_==-1)
    cen_cbn[i][noise] = -1
    end = time.time()
    print('Done DBSCAN {0} of {1} with eps {2} and min neighbours {3} - {4} seconds'.format(i+1,len(aeps),aeps[i],amin_samples[i],np.round(end-start,2)))
    pcount,plabs = membercount(db.labels_)
    bad = np.where(plabs==-1)
    if len(bad[0])>0:
        plabs = np.delete(plabs,bad[0][0])
        pcount = np.delete(pcount,bad[0][0])
    print('I found {0} out of {1} clusters'.format(len(plabs),numc))
datafile.close() 

spectra = specinfo.spectra
centers = clusters.centerdata[0]

plot = h5py.File('case7_{0}.hdf5'.format(clusters.timestamps[0].decode('UTF-8')),'w')

plot.attrs['sample'] = sample
plot.attrs['abundancefac'] = 0
plot.attrs['specfac'] = 0
plot.attrs['fullfit'] = [key.encode('utf8') for key in fullfitkeys]
plot.attrs['crossfit'] = crossfitatms
plot.attrs['spec_min'] = smin_samples
plot.attrs['spec_eps'] = seps
plot.attrs['abun_min'] = amin_samples
plot.attrs['abun_eps'] = aeps
plot['spec'] = spectra
plot['abun'] = abundances
plot['centers'] = centers
plot['labels_true'] = labels_true
plot['spec_labels_pred'] = spec_labels_pred
plot['abun_labels_pred'] = abun_labels_pred
plot['spec_cbn'] = spec_cbn
plot['abun_cbn'] = cen_cbn

tcount,tlabs = membercount(labels_true)
plot['true_size'] = tcount

neighbours = 20
d = distance_metrics(spectra)

plot['spec_true_sil_neigh{0}'.format(neighbours)] = d.silhouette(labels_true,k=neighbours)[0]

for i in range(len(seps)):
    efficiency, completeness, plabs, matchtlabs = efficiency_completeness(spec_labels_pred[i],
                                                                          labels_true,
                                                                          minmembers=1)
    pcount,plabs = membercount(spec_labels_pred[0])
    bad = np.where(plabs<0)
    if len(bad[0])>0:
        plabs = np.delete(plabs,bad[0][0])
        pcount = np.delete(pcount,bad[0][0])

    plot['spec_match_tlabs_eps{0}_min{1}'.format(seps[i],smin_samples[i])] = matchtlabs
    plot['spec_found_sil_eps{0}_min{1}_neigh{2}'.format(seps[i],smin_samples[i],neighbours) = d.silhouette(spec_labels_pred[i],k=neighbours)[0]
    plot['spec_eff_eps{0}_min{1}'.format(seps[i],smin_samples[i]) = efficiency
    plot['spec_com_eps{0}_min{1}'.format(seps[i],smin_samples[i]) = completeness
    plot['spec_found_size_eps{0}_min{1}'.format(seps[i],smin_samples[i]) = pcount

d = distance_metrics(abundances)

plot['spec_true_sil_neigh{0}'.format(neighbours)] = d.silhouette(labels_true,k=neighbours)[0]

for i in range(len(seps)):
    efficiency, completeness, plabs, matchtlabs = efficiency_completeness(abun_labels_pred[i],
                                                                          labels_true,
                                                                          minmembers=1)
    pcount,plabs = membercount(abun_labels_pred[0])
    bad = np.where(plabs<0)
    if len(bad[0])>0:
        plabs = np.delete(plabs,bad[0][0])
        pcount = np.delete(pcount,bad[0][0])

    plot['abun_match_tlabs_eps{0}_min{1}'.format(aeps[i],amin_samples[i])] = matchtlabs
    plot['abun_found_sil_eps{0}_min{1}_neigh{2}'.format(aeps[i],amin_samples[i],neighbours) = d.silhouette(abun_labels_pred[i],k=neighbours)[0]
    plot['abun_eff_eps{0}_min{1}'.format(aeps[i],amin_samples[i]) = efficiency
    plot['abun_com_eps{0}_min{1}'.format(aeps[i],amin_samples[i]) = completeness
    plot['abun_found_size_eps{0}_min{1}'.format(aeps[i],amin_samples[i]) = pcount


plot.close()


