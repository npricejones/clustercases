from case_template import *

# run parameters                                                               
nstars = 1e4 # number of stars                                                 
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
seps = np.array([0.5,0.6,0.7,0.8,0.9,1.0])
smin = np.array([2]*len(seps))
aeps = np.array([0.3,0.4,0.5,0.6,0.7])
amin = np.array([2]*len(aeps))
peps = np.array([0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
pmin = np.array([2]*len(peps))

combelem = ['Mg','Al','Si','S','K','Ca','Ni']

case = caserun(nstars=nstars,sample=sample,abundancefac=abundancefac,
                 spreadchoice=spreadchoice,specfac=specfac,centerfac=centerfac,
                 centerspr=spreads,genfn=choosestruct,
                 fullfitkeys=fullfitkeys,fullfitatms=fullfitatms,
                 crossfitkeys=crossfitkeys,crossfitatms=crossfitatms,
                 phvary=True,fitspec=True,case='12')
start = time.time()
case.clustering(case.specinfo.spectra,'spec',seps,smin,metric='precomputed',
                neighbours = 20,normeps=normeps)
case.clustering(case.abundances,'abun',aeps,amin,metric='precomputed',
                neighbours = 20,normeps=normeps)
case.clustering((case.abundances.T[abuninds(elem,combelem)]).T,'reda',aeps,amin,metric='precomputed',
                neighbours = 20,normeps=normeps)
toph = combine_windows(windows = tophats,combelem=elem,func=np.ma.any)
case.projspec(toph)
case.clustering(case.projectspec,'toph',seps,smin,metric='precomputed',
                neighbours = 20,normeps=normeps)
wind = combine_windows(windows = windows,combelem=combelem,func=np.ma.max)
case.projspec(wind)
case.clustering(case.projectspec,'wind',seps,smin,metric='precomputed',
                neighbours = 20,normeps=normeps)
case.reduction(reduct = PCA, n_components=20)
case.clustering(case.projectspec,'prin20',peps,pmin,metric='precomputed',
                 neighbours = 20,normeps=normeps)
case.reduction(reduct = PCA, n_components=10)
case.clustering(case.projectspec,'prin10',peps,pmin,metric='precomputed',
                 neighbours = 20,normeps=normeps)
case.reduction(reduct = PCA, n_components=50)
case.clustering(case.projectspec,'prin50',peps,pmin,metric='precomputed',
                 neighbours = 20,normeps=normeps)

end = time.time()
case.finish()
print('Finished desired clustering in {0} seconds'.format(end-start))
