from case_template import *

# run parameters                                                               
nstars = 7e4 # number of stars                                                 
sample='allStar_chemscrub_teffcut.npy' # APOGEE sample to draw from            
abundancefac = 1 # scaling factor for abundance noise                          
specfac = 0.01 # scaling factor for spectra noise                              
centerfac = 1
suff = 'H' # element denominator                                               
metric = 'precomputed' # metric for distances                                  
fullfitkeys = ['TEFF','LOGG'] # keys for the full fit                          
fullfitatms = []
crossfitkeys = []
crossfitatms = [26] # atomic numbers of cross terms                                                                    
spreadchoice = spreads # choose which abudance spreads to employ

# DBSCAN parameters                                                             
eps = np.array([0.7])
min_samples=np.array([3])

case = caserun(nstars=nstars,sample=sample,abundancefac=abundancefac,
                 spreadchoice=spreadchoice,specfac=specfac,centerfac=centerfac,
                 centerspr=spreads,genfn=choosestruct,
                 fullfitkeys=fullfitkeys,fullfitatms=fullfitatms,
                 crossfitkeys=crossfitkeys,crossfitatms=crossfitatms,
                 phvary=True,fitspec=True,case='20')
start = time.time()
case.clustering(case.specinfo.spectra,'spec',eps,min_samples,metric='euclidean',
                neighbours = 20,normeps=normeps)
case.clustering(case.abundances,'abun',eps,min_samples,metric='euclidean',
                neighbours = 20,normeps=normeps)
case.clustering((case.abundances.T[abuninds(elem,combelem)]).T,'reda',eps,min_samples,metric='euclidean',
                neighbours = 20,normeps=normeps)
toph = combine_windows(windows = tophats,combelem=elem,func=np.ma.any)
case.projspec(toph)
case.clustering(case.projectspec,'toph',eps,min_samples,metric='euclidean',
                neighbours = 20,normeps=normeps)
wind = combine_windows(windows = windows,combelem=combelem,func=np.ma.max)
case.projspec(wind)
case.clustering(case.projectspec,'wind',eps,min_samples,metric='euclidean',
                neighbours = 20,normeps=normeps)
case.reduction(reduct = PCA, n_components=10)
case.clustering(case.projectspec,'prin',eps,min_samples,metric='euclidean',
                 neighbours = 20,normeps=normeps)

end = time.time()
case.finish()
print('Finished desired clustering in {0} seconds'.format(end-start))
