from case_template import *

# run parameters                                                               
nstars = 1e4 # number of stars                                                 
sample='allStar_chemscrub_teffcut.npy' # APOGEE sample to draw from           
abundancefac = 1 # scaling factor for abundance noise                          
specfac = 0.01 # scaling factor for spectra noise                              
centerfac = 5
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

case16 = caserun(nstars=nstars,sample=sample,abundancefac=abundancefac,
                 spreadchoice=spreadchoice,specfac=specfac,
                 fullfitkeys=fullfitkeys,fullfitatms=fullfitatms,
                 crossfitkeys=crossfitkeys,crossfitatms=crossfitatms,
                 seps=seps,smin_samples=smin_samples,
                 aeps=aeps,amin_samples=amin_samples,
                 metric='precomputed',neighbours = 20,phvary=True,
                 fitspec=True,case='16',centerfac=centerfac,normeps=True)
