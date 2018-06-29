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

# DBSCAN parameters
amin_samples = np.array([2,3,4,5,6])
smin_samples = np.array([2,3])#,5,10])#,15,20,50])
asamples = len(amin_samples)
ssamples = len(smin_samples)
aeps = np.array([0.1,0.3,0.35,0.4,0.5])
seps = np.array([0.05,0.075,0.1])#,0.5,1.0])
amin_samples = np.tile(amin_samples,len(aeps))
smin_samples = np.tile(smin_samples,len(seps))
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
loggs = photospe['LOGG']

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

# Create fit terms

indeps = np.ones((mem,36))
indeps[:,1] = teffs-np.median(teffs) #T linear
indeps[:,2] = loggs-np.median(loggs) #logg linear
indeps[:,3] = (teffs-np.median(teffs))**2 #T squared
indeps[:,4] = (teffs-np.median(teffs))*(loggs-np.median(loggs)) #T logg
indeps[:,5] = (loggs-np.median(loggs))**2 # logg squared
indeps[:,6] = (teffs-np.median(teffs))*(chs-np.median(chs)) #T C
indeps[:,7] = (teffs-np.median(teffs))*(nhs-np.median(nhs)) #T N 
indeps[:,8] = (teffs-np.median(teffs))*(ohs-np.median(ohs)) #T O
indeps[:,9] = (teffs-np.median(teffs))*(nahs-np.median(nahs)) #T Na
indeps[:,10] = (teffs-np.median(teffs))*(mghs-np.median(mghs)) #T Mg
indeps[:,11] = (teffs-np.median(teffs))*(alhs-np.median(alhs)) #T Al
indeps[:,12] = (teffs-np.median(teffs))*(sihs-np.median(sihs)) #T Si
indeps[:,13] = (teffs-np.median(teffs))*(shs-np.median(shs)) #T S
indeps[:,14] = (teffs-np.median(teffs))*(khs-np.median(khs)) #T K
indeps[:,15] = (teffs-np.median(teffs))*(cahs-np.median(cahs)) #T Ca
indeps[:,16] = (teffs-np.median(teffs))*(tihs-np.median(tihs)) #T Ti
indeps[:,17] = (teffs-np.median(teffs))*(vhs-np.median(vhs)) #T V
indeps[:,18] = (teffs-np.median(teffs))*(mnhs-np.median(mnhs)) #T Mn
indeps[:,19] = (teffs-np.median(teffs))*(fehs-np.median(fehs)) # T Fe
indeps[:,20] = (teffs-np.median(teffs))*(nihs-np.median(nihs)) #T Ni 
indeps[:,21] = (loggs-np.median(loggs))*(chs-np.median(chs)) #g C 
indeps[:,22] = (loggs-np.median(loggs))*(nhs-np.median(nhs)) #g N  
indeps[:,23] = (loggs-np.median(loggs))*(ohs-np.median(ohs)) #g O 
indeps[:,24] = (loggs-np.median(loggs))*(nahs-np.median(nahs)) #g Na
indeps[:,25] = (loggs-np.median(loggs))*(mghs-np.median(mghs)) #g Mg
indeps[:,26] = (loggs-np.median(loggs))*(alhs-np.median(alhs)) #g Al
indeps[:,27] = (loggs-np.median(loggs))*(sihs-np.median(sihs)) #g Si
indeps[:,28] = (loggs-np.median(loggs))*(shs-np.median(shs)) #g S
indeps[:,29] = (loggs-np.median(loggs))*(khs-np.median(khs)) #g K
indeps[:,30] = (loggs-np.median(loggs))*(cahs-np.median(cahs)) #g Ca
indeps[:,31] = (loggs-np.median(loggs))*(tihs-np.median(tihs)) #g Ti
indeps[:,32] = (loggs-np.median(loggs))*(vhs-np.median(vhs)) #g V
indeps[:,33] = (loggs-np.median(loggs))*(mnhs-np.median(mnhs)) #g Mn
indeps[:,34] = (loggs-np.median(loggs))*(fehs-np.median(fehs)) #g Fe
indeps[:,35] = (loggs-np.median(loggs))*(nihs-np.median(nihs))#g Ni

findeps = create_indeps(mem,full=np.array([teffs,loggs]),  cross=np.array([chs,nhs,ohs,nahs,mghs,alhs,sihs,shs,khs,cahs,tihs,vhs,mnhs,fehs,nihs]))

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
 
"""
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

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('case8.pdf') as pdf:
    for p in range(len(spec_labels_pred)):
        size_distribution(spec_labels_pred[p],labels_true,
                          'eps = {0}, min ={1}'.format(seps[p],smin_samples[p]),
                          sizebins = np.linspace(1,1000,20))
        plt.savefig()
        plt.close()
    for p in range(len(cen_labels_pred)):
        size_distribution(cen_labels_pred[p],labels_true,
                          'eps = {0}, min ={1}'.format(aeps[p],amin_samples[p]),
                          sizebins = np.linspace(1,1000,20))
        plt.savefig()
        plt.close()
    
    
    plt.figure(figsize=(3, 3))
    plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
    plt.title('Page One')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    plt.rc('text', usetex=True)
    plt.figure(figsize=(8, 6))
    x = np.arange(0, 5, 0.1)
    plt.plot(x, np.sin(x), 'b-')
    plt.title('Page Two')
    pdf.attach_note("plot of sin(x)")  # you can add a pdf note to
                                       # attach metadata to a page
    pdf.savefig()
    plt.close()

    plt.rc('text', usetex=False)
    fig = plt.figure(figsize=(4, 5))
    plt.plot(x, x*x, 'ko')
    plt.title('Page Three')
    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
    plt.close()

    # We can also set the file's metadata via the PdfPages object:
    d = pdf.infodict()
    d['Title'] = 'Multipage PDF Example'
    d['Author'] = u'Jouni K. Sepp\xe4nen'
    d['Subject'] = 'How to create a multipage pdf file and set its metadata'
    d['Keywords'] = 'PdfPages multipage keywords author title subject'
    d['CreationDate'] = datetime.datetime(2009, 11, 13)
    d['ModDate'] = datetime.datetime.today()
    """
