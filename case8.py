from case_template import *

# run parameters                                                               
nstars = 1e4 # number of stars                                                 
sample='allStar_chemscrub.npy' # APOGEE sample to draw from                    
abundancefac = 1 # scaling factor for abundance noise                          
specfac = 0.01 # scaling factor for spectra noise                              
centerfac = 2
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
eps = np.array([0.07,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
min_samples = np.tile(min_samples,len(eps))
eps = np.repeat(eps,samples)

case8 = caserun(nstars=nstars,sample=sample,abundancefac=abundancefac,
                 spreadchoice=spreadchoice,specfac=specfac,centerfac=centerfac,
                 centerspr=spreads,genfn=choosestruct,
                 fullfitkeys=fullfitkeys,fullfitatms=fullfitatms,
                 crossfitkeys=crossfitkeys,crossfitatms=crossfitatms,
                 phvary=True,fitspec=True,case='8')
start = time.time()
case8.clustering(case8.specinfo.spectra,'spec',eps,min_samples,metric='precomputed',
                neighbours = 20,normeps=normeps)
case8.clustering(case8.abundances,'abun',eps,min_samples,metric='precomputed',
                neighbours = 20,normeps=normeps)
toph = combine_windows(windows = tophats,combelem=elem)
case8.projspec(toph)
case8.clustering(case8.projectspec,'toph',eps,min_samples,metric='precomputed',
                neighbours = 20,normeps=normeps)
end = time.time()
case8.finish()
print('Finished desired clustering in {0} seconds'.format(end-start))
