from case_template import *

# run parameters                                                               
nstars = 7e4 # number of stars                                                 
sample='allStar_chemscrub_teffcut_dr14.npy' # APOGEE sample to draw from       
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
jobs=320 
normeps=False
min_samples=np.array([3])

case = caserun(nstars=nstars,sample=sample,abundancefac=abundancefac,
                 spreadchoice=spreadchoice,specfac=specfac,centerfac=centerfac,
                 centerspr=spreads,genfn=choosestruct,
                 fullfitkeys=fullfitkeys,fullfitatms=fullfitatms,
                 crossfitkeys=crossfitkeys,crossfitatms=crossfitatms,
                 phvary=True,fitspec=True,case='100')
start = time.time()
case.clustering(case.specinfo.spectra,'spec',np.array([0.88]),min_samples,metric='euclidean',n_jobs=jobs,neighbours = 20,normeps=normeps)
case.clustering(case.abundances,'abun',np.array([0.367]),min_samples,metric='euclidean',n_jobs=jobs,neighbours = 20,normeps=normeps)
case.clustering((case.abundances.T[abuninds(elem,combelem)]).T,'reda',np.array([0.367]),min_samples,metric='euclidean',n_jobs=jobs,neighbours = 20,normeps=normeps)
case.gen_abundances(1,tingspr)
case.clustering(case.abundances,'tabn',np.array([0.117]),min_samples,metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
case.clustering((case.abundances.T[abuninds(elem,combelem)]).T,'trda',np.array([0.117]),min_samples,metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
case.reduction(reduct = PCA, n_components=30)
case.clustering(case.projectspec,'prin30',np.array([0.128]),min_samples,metric='euclidean',n_jobs=jobs,neighbours = 20,normeps=normeps)

end = time.time()
case.finish()
print('Finished desired clustering in {0} seconds'.format(end-start))
