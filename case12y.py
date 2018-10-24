from case_template import *

# run parameters                                                               
nstars = 5e4 # number of stars                                                 
sample= 'allStar_chemscrub_teffcut_dr14.npy'#'red_giant_teffcut_dr14.npy' # APOGEE sample to draw from            
abundancefac = 1 # scaling factor for abundance noise                          
specfac = 1e-2 # scaling factor for spectra noise                              
centerfac = 1
suff = 'H' # element denominator                                               
metric = 'precomputed' # metric for distances                                  
fullfitkeys = ['TEFF','LOGG'] # keys for the full fit                          
fullfitatms = []
crossfitkeys = []
crossfitatms = [26] # atomic numbers of cross terms                                                                   
spreadchoice = spreads # choose which abudance spreads to employ

# DBSCAN parameters                                                             

#min_samples = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,30,40,50])
min_samples = np.array([3])
samples = len(min_samples)
eps = np.array([0.01,0.02,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15,0.19,0.24,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
min_samples = np.tile(min_samples,len(eps))
eps = np.repeat(eps,samples)

jobs=8

combelem = ['Mg','Al','Si','S','K','Ca','Ni']

case = caserun()

case.makedata(nstars=nstars,sample=sample,abundancefac=abundancefac,volume=300,
                 spreadchoice=spreadchoice,specfac=specfac,centerfac=centerfac,
                 centerspr=spreads,genfn=choosestruct,
                 fullfitkeys=fullfitkeys,fullfitatms=fullfitatms,
                 crossfitkeys=crossfitkeys,crossfitatms=crossfitatms,
                 phvary=True,fitspec=True,case='12y',usecenters=True,add=True,
                 clsind=3.1)

start = time.time()

case.clustering(case.specinfo.spectra,'spec',eps,min_samples,metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
case.clustering(case.abundances,'abun',eps,min_samples,metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
#case.clustering((case.abundances.T[abuninds(elem,combelem)]).T,'reda',eps,min_samples,metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
case.gen_abundances(1,tingspr)
case.clustering(case.abundances,'tabn',eps,min_samples,metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
#case.clustering((case.abundances.T[abuninds(elem,combelem)]).T,'trda',eps,min_samples,metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
case.gen_abundances(1,leungspr)
case.clustering(case.abundances,'labn',eps,min_samples,metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
#case.clustering((case.abundances.T[abuninds(elem,combelem)]).T,'lrda',eps,min_samples,metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
#toph = combine_windows(windows = tophats,combelem=elem,func=np.ma.any)
#case.projspec(toph)
#case.clustering(case.projectspec,'toph',eps,min_samples,metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
#wind = combine_windows(windows = windows,combelem=combelem,func=np.ma.max)
#case.projspec(wind)
#case.clustering(case.projectspec,'wind',eps,min_samples,metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
#case.reduction(reduct = PCA, n_components=2)
#case.clustering(case.projectspec,'prin2',eps,min_samples,metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
#case.reduction(reduct = PCA, n_components=10)
#case.clustering(case.projectspec,'prin10',eps,min_samples,metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
case.reduction(reduct = PCA, n_components=30)
case.clustering(case.projectspec,'prin30',eps,min_samples,metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
#case.reduction(reduct = PCA, n_components=5)
#case.clustering(case.projectspec,'prin5',eps,min_samples,metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)

end = time.time()
case.finish()
print('Finished desired clustering in {0} seconds'.format(end-start))
