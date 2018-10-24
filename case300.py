from case_template import *

# run parameters                                                               
nstars = 5e4 # number of stars                                                 
sample='allStar_chemscrub_teffcut_dr14.npy' # APOGEE sample to draw from            
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

jobs=8

combelem = ['Mg','Al','Si','S','K','Ca','Ni']

case = caserun()

case.makedata(nstars=nstars,sample=sample,abundancefac=abundancefac,
                 spreadchoice=spreadchoice,specfac=specfac,centerfac=centerfac,
                 centerspr=spreads,genfn=choosestruct,
                 fullfitkeys=fullfitkeys,fullfitatms=fullfitatms,
                 crossfitkeys=crossfitkeys,crossfitatms=crossfitatms,
                 phvary=True,fitspec=True,case='300',usecenters=True,add=True)
start = time.time()

case.clustering(case.specinfo.spectra,'spec',[0.5],[3],metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
case.clustering(case.abundances,'abun',[0.24],[3],metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
case.gen_abundances(1,tingspr)
case.clustering(case.abundances,'tabn',[0.1],[3],metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
case.gen_abundances(1,leungspr)
case.clustering(case.abundances,'labn',[0.12],[3],metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)
case.reduction(reduct = PCA, n_components=30)
case.clustering(case.projectspec,'prin30',[0.12],[3],metric='precomputed',n_jobs=jobs,neighbours = 20,normeps=normeps)

end = time.time()
case.finish()
print('Finished desired clustering in {0} seconds'.format(end-start))
