''' Present a scatter plot with linked histograms on both axes.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve selection_histogram.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/selection_histogram
in your browser.
'''
import os
import h5py
import glob
import copy
import numpy as np
import warnings
from clustering_stats import *

from bokeh.layouts import row, column, widgetbox
from bokeh.models import BoxSelectTool, LassoSelectTool, Spacer,CustomJS, Legend
from bokeh.models.glyphs import Circle
from bokeh.models.widgets import Toggle,RadioButtonGroup,AutocompleteInput,Tabs, Panel, Select, Button, TextInput
from bokeh.plotting import figure, curdoc, ColumnDataSource, reset_output
from bokeh.io import show, export_svgs
from bokeh import events

# default neighbours being used for silhouette coefficient calculation
neighbours = 20


# tools to include on bokeh plot
TOOLS="pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

cwd = os.getcwd()

resultpath = os.getenv('CLUSTERCASES', cwd)

# Types of data that clustering was run on

typenames = {'spec':'spectra','abun':'abundances','reda':'reduced abundances','toph':'tophat windows','wind':'windows','prin':'principal components'}
nametypes = {'spectra':'spec','abundances':'abun','reduced abundances':'reda','tophat windows':'toph','windows':'wind','principal components':'prin'}

#             orange     purple     blue        red       green     dark green
colorlist = ["#F98D20", "#904C77", "#79ADDC", "#ED6A5A", "#B1EF73","#6CAE75"] 

typecolor = {'spec':"#F98D20",'abun':"#904C77",'reda':"#6CAE75",'toph':"#79ADDC",'wind':"#ED6A5A",'prin':"#B1EF73"}

zp = {'Efficiency':5e-3,'Completeness':5e-3,'Found Silhouette':5e-3,'Matched Silhouette':5e-3,'Found Size':0.5,'Matched Size':0.5}
lzp=1e-3
plot_eps = 0.5
padfac = 0.1

def findextremes(x,y,pad=0.1):
    """
    Try to find the limits of the dataset. If you can't because it's empty, 
    choose arbitrary values.

    x:      an array of data
    y:      another array of data
    pad:    fraction with which to pad extremes

    Returns the limits of each axis and ultimate extremes.
    """
    try:
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
    except ValueError:
        xmin = 0.5
        xmax = 1
        ymin = 0.5
        ymax = 1
    # Add padding
    xmin -= padfac*pad*xmax
    xmax += pad*xmax
    ymin -= padfac*pad*ymax
    ymax += pad*ymax
    minlim = np.min([xmin,ymin])
    maxlim = np.max([xmax,ymax])
    return (xmin,xmax,ymin,ymax),[minlim,maxlim]

def create_case_list(direc='.'):
    caselist = glob.glob('{0}/*.hdf5'.format(direc))
    cases = np.unique(np.array([i.split('_')[0].split('case')[-1] for i in caselist])).astype('int')
    cases.sort()
    cases = cases.astype('str')
    return cases

def create_time_list(case,direc='.'):
    timelist = glob.glob('{1}/case{0}*.hdf5'.format(case,direc))
    times = np.array([i.split('_')[1].split('.hdf5')[0] for i in timelist])[::-1]
    return times

def set_colors(objct):
    """
    Attach some colors

    Returns list of colours
    """
    objct.bcolor = "#FFF7EA" #cream
    objct.unscolor = "#4C230A" #dark brown
    objct.maincolor = "#A53F2B" #dark red
    objct.histcolor = "#F6BD60" #yellow
    objct.outcolor = "#280004" #dark red black
    objct.colorlist = colorlist
    return [objct.bcolor,objct.unscolor,objct.outcolor,
            objct.histcolor,objct.maincolor]

class read_results(object):

    """
    Functions to gather data from an hdf5 file.

    """

    def __init__(self, datatype = 'spec', case = 7, 
                 timestamp = '2018-07-06.14.28.11.782906'):
        """
        datatype:       type of data to gather (spec, abun, toph or wind)
        case:           case number of file
        timestamp:      timestamp of file

        returns None
        """
        self.dtype = datatype
        self.case = case
        self.timestamp = timestamp

    def read_base_data(self,datatype=None,case=None,timestamp=None,
                       neighbours=20):
        """
        Set global parameters and opens case file.

        datatype:       type of data to gather (spec, abun, toph or wind)
        case:           case number of file - default is from init
        timestamp:      timestamp of file - default is from init
        neighbours:     number of neighbours used in silhouette calculation
        """
        if not datatype:
            datatype = self.dtype
        # open the h5py file
        self.data = h5py.File('case{0}_{1}.hdf5'.format(self.case,
                                                        self.timestamp),'r+')
        # set the upper limit when plotting the number of clusters
        self.maxmem = 1
        # aquire defaults
        if case:
            self.case = case
        if timestamp:
            self.timestamp = timestamp
        # generate list of all allowed data types
        alldtypes = np.unique(np.array([i.split('_')[0] for i in list(self.data.keys())]))
        special = np.where((alldtypes=='true') | (alldtypes=='labels'))
        if len(special[0])>0:
            alldtypes = np.delete(alldtypes,special[0])
        self.alldtypes = []
        for dtype in alldtypes:
            self.alldtypes.append(typenames[dtype])

        # read in the true cluster sizes for this file
        self.tsize = self.data['true_size'][:]

        # read in datatype specific data
        self.read_dtype_data(datatype=datatype)
        
    def read_dtype_data(self,datatype=None):
        """
        Reads in datatype specifc data

        datatype:       type of data to gather - default is from init
        """
        if not datatype:
            datatype = self.dtype

        # read in silhouette coefficients of true clusters
        self.tsil = self.data['{0}_true_sil_neigh{1}'.format(datatype,
                                                             neighbours)][:]
        # scrub nans from silhouette coefficients
        self.tsil[np.isnan(self.tsil)]=-1

        # read in predicted cluster labels across all parameter choices
        self.labels_pred = self.data['{0}_labels_pred'.format(datatype)][:]

        # create arrays to store cluster properties
        self.numcs = []
        self.numms = []

        # track 'good index', the index where there are clusters so statistics
        # can be plotted
        self.goodind = 0

        # cycle through all parameter choices
        for row in range(self.labels_pred.shape[0]):
            # count the number of clusters and their members
            labcount,labs = membercount(self.labels_pred[row])
            # excise outlier cluster
            bad = np.where(labs==-1)
            if len(bad[0])>0:
                labs = np.delete(labs,bad[0][0])
                labcount = np.delete(labcount,bad[0][0])

            # store attributes
            self.numcs.append(len(labs))
            self.numms.append(labcount)

        self.numcs = np.array(self.numcs)
        # only plot if there's enough points (3 established by trial and error)
        self.goodinds = np.where(self.numcs > 3)
        
        # check whether any of the parameter choices have enough clusters 
        # to plot are store result in allbad parameter
        self.allbad = False
        # if there are parameters with clusters found, use the first one by default
        if len(self.goodinds[0]) > 0:
            self.goodind = self.goodinds[0][0]
        elif len(self.goodinds[0])==0:
            self.allbad = True

        # read in properties of the runs
        self.min_samples = self.data.attrs['{0}_min'.format(datatype)][:]
        self.eps = self.data.attrs['{0}_eps'.format(datatype)][:]

        # choose the run of interest, and globally note its parameters
        self.epsval = self.eps[self.goodind]
        self.minval = self.min_samples[self.goodind]

        # create string lists of the parameters for summary plot labels and
        # the drop down menu for parameter selection
        self.paramchoices = []
        self.ticklabels = []
        for i in range(len(self.eps)):
            self.paramchoices.append('eps={0}, min={1}'.format(self.eps[i],self.min_samples[i]))
            self.ticklabels.append('{0}, {1}'.format(self.eps[i],self.min_samples[i]))
        self.paramlist = list(np.array(self.paramchoices)[self.goodinds])

    def generate_average_stats(self,minmem=1,update=False):
        """
        Find average properties across all data types in a run

        minmem:     minimum number of members a cluster needs to be included 
                    in the statistics
        update:     if true, do not reassign source dictionary

        returns None
        """

        # Track what datatype we were using to start
        vintdtype = copy.deepcopy(self.dtype)

        # Create a master list of labels in case different datatypes 
        # had different parameters 
        labmaster = []

        # create and flatten label list
        for d,dtype in enumerate(self.alldtypes):
            dtype = nametypes[dtype]
            self.read_dtype_data(datatype=dtype)
            labmaster.append(self.ticklabels)

        labmaster = np.unique(np.array([item for sublist in labmaster for item in sublist]))
        
        # create list of xvalues to find where ticks should be plotted
        xvals = np.arange(len(labmaster))

        # for each data type, create array to hold result of the runs
        for d,dtype in enumerate(list(nametypes.keys())):
            dtype = nametypes[dtype]

            # Initialize arrays
            effs = np.zeros(len(labmaster))
            coms = np.zeros(len(labmaster))
            fsil = -np.ones(len(labmaster))
            numc = 0.01*np.ones(len(labmaster))

            if typenames[dtype] in self.alldtypes:
                # Read in file data
                self.read_dtype_data(datatype=dtype)
                # cycle through epsilon values for this run
                for e,eps in enumerate(self.eps):
                    sizes = self.numms[e]
                    # Read in specific run data if it exists for current datatype
                    try:
                        self.read_run_data(eps=eps,min_sample=self.min_samples[e],update=True,datatype=dtype)
                        # If more than three clusters found, calculate averages
                        if len(self.eff) > 3:
                            # Find the number of sufficiently large cluster
                            vals = sizes >= minmem
                            if np.sum(vals) > 0:
                                vals = np.where(vals)
                                match = np.where(labmaster=='{0}, {1}'.format(eps,self.min_samples[e]))
                                numc[match] = len(sizes[vals])
                                effs[match] = np.mean(self.eff[vals])
                                coms[match] = np.mean(self.com[vals])
                                fsil[match] = np.mean(self.fsil[vals])
                            self.maxmem = np.max([self.maxmem,np.max(numc)])
                    except KeyError:
                        pass
            # If datatypes do not exist, move points out of summary plot range
            if typenames[dtype] not in self.alldtypes:
                effs -= 1
                coms -= 1
                fsil -= 1
            # Calculate the true number of clusters above a given limit
            tnumc = np.array([len(self.tsize[self.tsize>minmem])]*len(labmaster))
            self.maxmem = np.max([self.maxmem,tnumc[0]])
            # If the true number of clusters above the limit is None, move out of plot range
            tnumc[tnumc < 1] = 0.01
            # Create dictionary for plotting
            statsource = {'params':labmaster,'numc':numc,
                               'avgeff':effs,'avgcom':coms,
                               'avgfsi':fsil,
                               'xvals':xvals,
                               'tnumc':tnumc}
            # Add ColumnDataSource object to class
            setattr(self,'{0}_statsource'.format(dtype),ColumnDataSource(statsource))

            # Add ColumnDataSource object to source dictionary
            if 'sourcedict' not in dir(self):
                self.sourcedict={}
            if not update:
                self.sourcedict['{0}source'.format(dtype)] = getattr(self,'{0}_statsource'.format(dtype))
            self.sourcedict['new{0}source'.format(dtype)] = getattr(self,'{0}_statsource'.format(dtype))
        # Reset to original datatype and redo the read
        self.dtype = vintdtype
        self.read_dtype_data()
        self.read_run_data(eps=self.epsval,min_sample=self.minval,update=False)

    def read_run_data(self,eps=None,min_sample=None,update=False,datatype=None):
        """
        Read in arrays for a specifc fun.

        eps:            DBSCAN parameter epsilon defining regions of high density
        min_sample:     DBSCAN parameter minimum samples in a clusters
        update:         Boolean key that specifies whether to change source key in self.sourcedict
        datatype:       Allows an override of current datatype


        Returns None
        """
        # If no datatype set, use default
        if not datatype:
            datatype = self.dtype
        # If no epsilon set, use default
        if not eps:
            eps = self.epsval
        # If no min_sample, use default
        if not min_sample:
            min_sample = self.minval

        # Find matches to found clusters
        self.matchtlabs = self.data['{0}_match_tlabs_eps{1}_min{2}'.format(datatype,eps,min_sample)][:]
        # Use matched clusters to find matched sizes and silhouette
        if len(self.matchtlabs) > 0:
            self.msil = self.tsil[self.matchtlabs]
            self.msize = self.tsize[self.matchtlabs]
        # If there were no clusters, store empty arrays
        elif len(self.matchtlabs) == 0:
            self.msil = np.array([])
            self.msize = np.array([])
        # Read in remaining metrics for this run
        self.fsil = self.data['{0}_found_sil_eps{1}_min{2}_neigh{3}'.format(datatype,eps,min_sample,neighbours)][:]
        self.eff = self.data['{0}_eff_eps{1}_min{2}'.format(datatype,eps,min_sample)][:]
        self.com = self.data['{0}_com_eps{1}_min{2}'.format(datatype,eps,min_sample)][:]
        self.fsize = self.data['{0}_found_size_eps{1}_min{2}'.format(datatype,eps,min_sample)][:]
        
        # Assign the total number of clusters
        self.numc = len(self.fsize)
        
        # Scrub nans
        self.msil[np.isnan(self.msil)] = -1
        self.fsil[np.isnan(self.fsil)] = -1
        self.eff[np.isnan(self.eff)] = 0
        self.com[np.isnan(self.com)] = 0

        # Create source dictionary for scatter plot
        self.datadict = {'Efficiency':self.eff,'Completeness':self.com,
                         'Found Silhouette':self.fsil,'Matched Silhouette':self.msil,
                         'Found Size':self.fsize,'Matched Size':self.msize}
        self.source=ColumnDataSource(data=self.datadict)

        # Add ColumnDataSource object to source dictionary
        if 'sourcedict' not in dir(self):
            self.sourcedict={}
        if not update:
            self.sourcedict['source'] = self.source
        self.sourcedict['newsource'] = self.source

class display_single(read_results):
    """
    Creates scatter plot and histograms of data for a specific run
    """

    def __init__(self, datatype = 'spec', case = 7, 
                 timestamp = '2018-06-29.20.20.13.729481', pad = 0.1, 
                 sqside = 460,tools=TOOLS):

        """
        Sets basic variables
        ddatatype:      type of data to gather (spec, abun, toph or wind)
        case:           case number of file
        timestamp:      timestamp of file
        pad:            amount by which to pad axis limits from extremes of the data
        sqside:         size of main scatter plot
        tools:          tools to show in bokeh toolbar

        Returns None      
        """
        # Get data set up
        read_results.__init__(self,datatype=datatype,case=case,
                              timestamp=timestamp)
        self.sqside=sqside
        self.tools=tools
        self.pad = pad

        # Set plot colours
        set_colors(self)
        # Create plots
        self.layout_plots()
        
# MODIFY FOR OLD SOURCE

    def JScallback(self):
        """
        Makes custom JavaScript callback from bokeh so you can easily swap source dictionaries.
        """ 

        self.callbackstr =''         

        keylist = list(self.sourcedict.keys())
        keylist.remove('button')
        # Initizlize variables
        for key in list(keylist):
            self.callbackstr += """
var v{0} = {0}.data;""".format(key)

        # Update contents of the variables
        for key in list(keylist):
            if 'new' not in key:
                self.callbackstr += """
for (key in vnew{0}) {{
    v{0}[key] = [];
    for (i=0;i<vnew{0}[key].length;i++){{
    v{0}[key].push(vnew{0}[key][i]);
    }}
}}""".format(key)
    
        # Push changes to the variables
        for key in list(keylist):
            if 'new' not in key:
                self.callbackstr += """
{0}.change.emit();""".format(key)
        self.callbackstr += """
button.label = 'I do nothing until you select new run info above';
button.button_type = 'warning';"""

    def layout_plots(self):
        """
        Initializes plots and lays them out.
        """
        # Read in base data
        self.read_base_data()

        # Check if any of the runs found any clusters
        if self.allbad:
            print("Didn't find any clusters for any parameter choices with {0} this run".format(typenames[self.dtype]))
        # If some clusters were found, continue
        elif not self.allbad:
            # Load in data for the specific run using the first set of parameters with clusters found
            self.read_run_data()
            # Generate scatter plots
            self.center_plot()
            # Generate histograms
            self.histograms()
            # Create buttons
            self.buttons()

            # Set up layout
            buttons = column(widgetbox(self.selectcase,width=350,height=30),
                             widgetbox(self.selecttime,width=350,height=30),
                             widgetbox(self.selectdtype,width=350,height=30),
                             widgetbox(self.selectparam,width=350,height=30),
                             widgetbox(self.loadbutton,width=350,height=30),
                             widgetbox(self.toggleline,width=350,height=30),
                             widgetbox(self.xradio,width=350),
                             widgetbox(self.yradio,width=350))

            mainplot = row(buttons,
                           Tabs(tabs=self.panels,width=self.sqside+20),
                           column(Spacer(width=210,height=80),
                                  self.p_found_size,
                                  self.p_matched_size))


            histplot = row(self.p_efficiency,
                           self.p_completeness,
                           self.p_found_silhouette,
                           self.p_matched_silhouette)
            # Here's where you decide the distribution of plots
            self.layout = column(mainplot,histplot)

            # Push data to documents
            curdoc().add_root(self.layout)
            curdoc().title = "DBSCAN results"

    def center_plot(self,xlabel=None,ylabel=None):
        """
        Create central scatter plot

        xlabel:     key to get xdata from self.source
        ylabel:     key to get ydata from self.source

        Returns None
        """

        # Create list to hold panels with different axis scales
        self.panels = []
        # If no x/ylabel, use defaults
        if not xlabel:
            xlabel = 'Efficiency'
        if not ylabel:
            ylabel = 'Completeness'
        x = self.source.data[xlabel]
        y = self.source.data[ylabel]
        # Determine axis limits
        axlims,lineparams = findextremes(x,y,pad=self.pad)
        xmin,xmax,ymin,ymax=axlims
        minlim,maxlim = lineparams

        # Find special limits for log scale
        if minlim < 0:
            lminlim = lzp
        else:
            lminlim = minlim

        # LINEAR TAB
        self.p1 = figure(tools=TOOLS, plot_width=self.sqside, 
                         plot_height=self.sqside, min_border=10, 
                         min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left',
                         x_axis_label=xlabel,y_axis_label=ylabel,
                         x_axis_type='linear',y_axis_type='linear',
                         x_range=(xmin,xmax),y_range=(ymin,ymax))

        # set background color and update selection tool behaviour to be less frenetic
        self.p1.background_fill_color = self.bcolor
        self.p1.select(BoxSelectTool).select_every_mousemove = False
        self.p1.select(LassoSelectTool).select_every_mousemove = False

        # Create scatter points
        self.r1 = self.p1.scatter(x=xlabel, y=ylabel, source=self.source, 
                                  size=3, color=self.maincolor, alpha=0.6)

        # Initialize one-to-one line
        self.l1 = self.p1.line(lineparams,lineparams,
                               color=self.outcolor)

        # Update colour of unselected points
        self.r1.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        # Add panel to list
        self.panels.append(Panel(child=self.p1,title='linear'))


        # SEMILOGX TAB

        # Find log-appropriate axis limits
        if xmin < 0:
            slxmin = zp[xlabel]
        else:
            slxmin = xmin
        self.p2 = figure(tools=TOOLS, plot_width=self.sqside, 
                         plot_height=self.sqside, min_border=10, 
                         min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left',
                         x_axis_label=xlabel,y_axis_label=ylabel,
                         x_axis_type='log',y_axis_type='linear',
                         x_range=(slxmin,xmax),y_range=(ymin,ymax))

        # set background color and update selection tool behaviour to be less frenetic
        self.p2.background_fill_color = self.bcolor
        self.p2.select(BoxSelectTool).select_every_mousemove = False
        self.p2.select(LassoSelectTool).select_every_mousemove = False

        # Create scatter points
        self.r2 = self.p2.scatter(x=xlabel,y=ylabel, source=self.source, 
                                  size=3, color=self.maincolor, alpha=0.6)
    
        # Initialize one-to-one line
        self.l2 = self.p2.line(np.logspace(np.log10(lminlim),
                                           np.log10(maxlim),100),
                               np.logspace(np.log10(lminlim),
                                           np.log10(maxlim),100),
                               color=self.outcolor)

        # Update color of unselected points
        self.r2.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        # Add panel to list
        self.panels.append(Panel(child=self.p2,title='semilogx'))


        # SEMILOGY TAB

        # Find log-appropriate axis limits
        if ymin < 0:
            slymin = zp[ylabel]
        else:
            slymin = ymin
        self.p3 = figure(tools=TOOLS, plot_width=self.sqside, 
                         plot_height=self.sqside, min_border=10, 
                         min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left', 
                         x_axis_label=xlabel, y_axis_label=ylabel, 
                         x_axis_type='linear', y_axis_type='log', 
                         x_range=(xmin,xmax), y_range=(slymin,ymax))

        # set background color and update selection tool behaviour to be less frenetic
        self.p3.background_fill_color = self.bcolor
        self.p3.select(BoxSelectTool).select_every_mousemove = False
        self.p3.select(LassoSelectTool).select_every_mousemove = False

        # Create scatter points
        self.r3 = self.p3.scatter(x=xlabel,y=ylabel, source=self.source, 
                                  size=3, color=self.maincolor, alpha=0.6)
  
        # Initialize one-to-one line
        self.l3 = self.p3.line(np.logspace(np.log10(lminlim),
                                           np.log10(maxlim),100),
                               np.logspace(np.log10(lminlim),
                                           np.log10(maxlim),100),
                               color=self.outcolor)

        # Update color of unselected points
        self.r3.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        # Add panel to list
        self.panels.append(Panel(child=self.p3,title='semilogy'))


        # LOG PLOT
        self.p4 = figure(tools=TOOLS, plot_width=self.sqside, 
                         plot_height=self.sqside, min_border=10, 
                         min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left', 
                         x_axis_label=xlabel, y_axis_label=ylabel, 
                         x_axis_type='log', y_axis_type='log', 
                         x_range=(slxmin,xmax), y_range=(slymin,ymax))

        # set background color and update selection tool behaviour to be less frenetic
        self.p4.background_fill_color = self.bcolor
        self.p4.select(BoxSelectTool).select_every_mousemove = False
        self.p4.select(LassoSelectTool).select_every_mousemove = False

        # Create scatter points
        self.r4 = self.p4.scatter(x=xlabel,y=ylabel, source=self.source, 
                                  size=3, color=self.maincolor, alpha=0.6)

        # Initialize one-to-one line
        self.l4 = self.p4.line([lminlim,maxlim],[lminlim,maxlim],
                               color=self.outcolor)

        # Update color of unselected points
        self.r4.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        # Add panel to list
        self.panels.append(Panel(child=self.p4,title='log'))

        # Make lists for simplified iteration
        self.ps = [self.p1,self.p2,self.p3,self.p4]
        self.rs = [self.r1,self.r2,self.r3,self.r4]
        self.ls = [self.l1,self.l2,self.l3,self.l4]

    def make_hist(self,key,bins=20,x_range=(),xscale='linear',
                  yscale='linear',background=[], update=False):
        """
        Create a histogram

        key:        points to array to histogram in self.source
        bins:       number of bins in the histogram
        x_range:    bounds of the histogram
        xscale:     scale of x-axis
        yscale:     scale of y-axis
        background: possible background histogram array
        update:     boolean to specify whether to plot the histogram or just get data

        Returns None
        """
        # Convert key string into proper key for source
        hist_name = key.lower().replace(' ','_')
        arr = self.source.data[key] 
        # Store array being used to generate the histogram ***
        setattr(self,'arr_{0}'.format(hist_name),arr)
        # If background requested, generate it
        hist_max = 0
        if background != []:
            edgeset = True
            backhist,edges = np.histogram(background,bins=bins)
            backhist = backhist.astype('float')
            setattr(self,'bhist_{0}'.format(hist_name),backhist)
            backmax = np.max(backhist)*1.1
            # If the background histogram is taller than the foreground, update
            # the max height accordingly
            if backmax > hist_max:
                hist_max = backmax

            # Generate histogram
            hist, edges = np.histogram(arr, bins=edges)
        elif background == []:
            # Generate histogram
            hist, edges = np.histogram(arr, bins=bins)
        hist = hist.astype('float')
        # Store histogram and the edges ***
        setattr(self,'hist_{0}'.format(hist_name),hist)
        setattr(self,'edges_{0}'.format(hist_name),edges)
        # Create and store zero height array ***
        zeros = np.zeros(len(edges)-1)
        setattr(self,'zeros_{0}'.format(hist_name),zeros)
        # Find the upper limit of the histogram and store it ***
        if max(hist)*1.1 > hist_max:
            hist_max = max(hist)*1.1
        setattr(self,'hist_max_{0}'.format(hist_name),hist_max)
        ymax = hist_max
        # If no x_range is set, use the extent of the data
        if x_range==():
            x_range = (np.min(edges),np.max(edges))

        # Choose a y range minimum based on yscale
        if yscale=='linear':
            ymin = 0
        elif yscale=='log':
            ymin=plot_eps+plot_eps/2.
            # Scrub zeros from the data and update the attribute
            hist[hist < plot_eps] = plot_eps
            setattr(self,'hist_{0}'.format(hist_name),hist)
            if background != []:
                backhist[backhist < plot_eps] = plot_eps
                setattr(self,'bhist_{0}'.format(hist_name),backhist)
        # Store y range minimum ***
        setattr(self,'ymin_{0}'.format(hist_name),ymin)

        # Create and store column data source for histogram data
        histsource = {'mainhist':hist,'left':edges[:-1],'right':edges[1:],
                      'bottom':ymin*np.ones(len(hist)),'zeros':zeros,'selected':zeros}
        if background != []:
            histsource['backhist'] = backhist
        histsource = ColumnDataSource(data=histsource)
        setattr(self,'hsource_{0}'.format(hist_name),histsource)

        # If plot requested, make it
        if not update:
            p = figure(toolbar_location=None, plot_width=210, plot_height=200,
                        x_range=x_range,y_range=(ymin, hist_max), min_border=10, 
                        min_border_left=20, y_axis_location="right",
                        x_axis_label=key,x_axis_type=xscale,
                        y_axis_type=yscale)
            # Store the plot for future updates
            setattr(self,'p_{0}'.format(hist_name),p)
            p.xgrid.grid_line_color = None
            #pt.yaxis.major_label_orientation = np.pi/4
            p.background_fill_color = self.bcolor

            # Add plots in order of their zstack
            if background != []:
                bghist = p.quad(bottom='bottom', left='left', right='right', 
                                top='backhist', 
                                source = getattr(self,'hsource_{0}'.format(hist_name)), 
                                color=self.unscolor, line_color=self.outcolor)
            mnhist = p.quad(bottom='bottom', left='left', right='right', 
                         top='mainhist', alpha=0.7,
                         source = getattr(self,'hsource_{0}'.format(hist_name)),
                         color=self.histcolor, line_color=self.outcolor)
            
            # This histogram will update to show the distribution of the user selection in the
            # scatter plot, and as such, must be globally stored
            h1 = p.quad(bottom='bottom', left='left', right='right', 
                         top='selected', alpha=0.6, 
                         source = getattr(self,'hsource_{0}'.format(hist_name)),
                         color=self.maincolor,line_color=None)
            setattr(self,'h_{0}'.format(hist_name),h1)


    def histograms(self,nbins=20,update=False):
        # Make a histogram for each metric and the sizes

        # Efficiency
        self.make_hist('Efficiency',bins=np.linspace(0,1,nbins),
                  x_range=(0,1),yscale='log',update=update)

        # Completeness
        self.make_hist('Completeness',bins=np.linspace(0,1,nbins),
                  x_range=(0,1),yscale='log',update=update)

        # Found Silhouette Coefficient
        self.make_hist('Found Silhouette',bins=np.linspace(-1,1,nbins),
                  x_range=(-1,1),yscale='log',update=update)

        # Matched Silhouette Coefficient
        self.make_hist('Matched Silhouette',bins=np.linspace(-1,1,nbins),
                  x_range=(-1,1),yscale='log',background=self.tsil,
                  update=update)

        # Find the upper bin bound for the size plots
        try:
            self.maxsize = np.max(np.array([np.max(self.source.data['Found Size']),
                                            np.max(self.source.data['Matched Size']),
                                            np.max(self.tsize)]))
            self.maxsize = np.log10(self.maxsize)
        except ValueError:
            self.maxsize=10

        # Found Cluster Sizes
        self.make_hist('Found Size',bins=np.logspace(0,self.maxsize,nbins),
                  xscale='log',yscale='log',background=self.tsize,
                  update=update)

        # Matched Cluster Sizes
        self.make_hist('Matched Size',bins=np.logspace(0,self.maxsize,nbins),
                  xscale='log',yscale='log',background=self.tsize,
                  update=update)

        # If not updating, make a list for easy recovery and update the source dict
        if not update:
            self.histlist = [self.h_efficiency,self.h_completeness,
                             self.h_found_silhouette,
                             self.h_matched_silhouette,
                             self.h_found_size,self.h_matched_size]

            self.sourcedict['heff'] = self.hsource_efficiency
            self.sourcedict['hcom'] = self.hsource_completeness
            self.sourcedict['hfsi'] = self.hsource_found_silhouette
            self.sourcedict['hmsi'] = self.hsource_matched_silhouette
            self.sourcedict['hfsz'] = self.hsource_found_size
            self.sourcedict['hmsz'] = self.hsource_matched_size

        # Update the source dict with the new CDS objects
        self.sourcedict['newheff'] = self.hsource_efficiency
        self.sourcedict['newhcom'] = self.hsource_completeness
        self.sourcedict['newhfsi'] = self.hsource_found_silhouette
        self.sourcedict['newhmsi'] = self.hsource_matched_silhouette
        self.sourcedict['newhfsz'] = self.hsource_found_size
        self.sourcedict['newhmsz'] = self.hsource_matched_size



    def buttons(self):
        """
        Creates buttons that manipulate the plots
        """
        cases = create_case_list()
        times = create_time_list(self.case)

        # Create list of possible axis for scatter plot
        self.labels = list(self.source.data.keys())

        self.xradio = RadioButtonGroup(labels=self.labels, active=0,name='x-axis')
        self.yradio = RadioButtonGroup(labels=self.labels, active=1,name='y-axis')
        self.xradio.on_click(self.updateallx)
        self.yradio.on_click(self.updateally)

        # Create drop down menu for possible cases
        self.selectcase = Select(title='case',value=self.case,options=list(cases))
        self.selectcase.on_change('value',self.updatecase)

        # Create drop down menu for possible timestamps
        self.selecttime = Select(title='timestamp',value=self.timestamp,options=list(times))
        self.selecttime.on_change('value',self.updatetime)

        # Create drop down menu for possible data types
        self.selectdtype = Select(title='data type',value='spectra',options=self.alldtypes)
        self.selectdtype.on_change('value',self.updatedtype)
        
        # Create drop down menu for possible DBSCAN parameters
        self.selectparam = Select(title="parameter values", value=self.paramchoices[self.goodind], 
                           options=self.paramlist)
        self.selectparam.on_change('value',self.updateparam)

        # Create button to actually push results of new data to plot
        self.loadbutton = Button(label='I do nothing until you select new run info above', button_type='warning')
        self.sourcedict['button'] = self.loadbutton
        self.JScallback()
        self.loadbutton.callback = CustomJS(args=self.sourcedict,code=self.callbackstr)
        
        # Create toggle for one-to-one visibility
        code = '''\
        object1.visible = toggle.active
        object2.visible = toggle.active
        object3.visible = toggle.active
        object4.visible = toggle.active
        '''
        linecb = CustomJS.from_coffeescript(code=code, args={})
        self.toggleline = Toggle(label="One-to-one line", button_type="default", active=True,callback=linecb)
        linecb.args = {'toggle': self.toggleline, 'object1': self.l1, 'object2': self.l2, 'object3': self.l3, 'object4': self.l4}

        # Add callbacks to update histograms with data selected from scatter plot
        self.r1.data_source.on_change('selected', self.updatetophist)
        self.r2.data_source.on_change('selected', self.updatetophist)
        self.r3.data_source.on_change('selected', self.updatetophist)
        self.r4.data_source.on_change('selected', self.updatetophist)

        # Add callbacks so that when the central panels are rest the histograms are too
        self.p1.on_event(events.Reset,self.resetplots)
        self.p2.on_event(events.Reset,self.resetplots)
        self.p3.on_event(events.Reset,self.resetplots)
        self.p4.on_event(events.Reset,self.resetplots)

    def resetplots(self,attrs):
        """
        Resets all histograms to initial states

        attrs:  bokeh mandated arg, does nothing

        Return None
        """
        # Update the axis limits so that resetting to does not go all the way back to the
        # initial axis limits (which may have been for different data)
        self.updateaxlim()
        # Check whether I have a set of keys to iterate over - if not add them
        if 'histkeys' not in dir(self):
            histkeys = self.datadict.keys()
            self.histkeys = np.array([key.lower().replace(' ','_') for key in histkeys])
        for key in self.histkeys:
            h = getattr(self,'h_{0}'.format(key))
            # Update the top of the selection histogram to zeros
            h.glyph.top = 'zeros'


    def updatetophist(self, attr, old, new):
        """
        Updates the tops of the selection histogram. Call back for scatter.on_change

        attr:       bokeh mandated arg, does nothing
        old:        bokeh mandated arg, does nothing
        new:        bokeh mandated arg, provides indices of data to histogram

        Returns None
        """

        # Check whether I have a set of keys to iterate over - if not add them
        if 'histkeys' not in dir(self):
            histkeys = self.datadict.keys()
            self.histkeys = np.array([key.lower().replace(' ','_') for key in histkeys])

        # Get inds to know what part of the data to histogram
        inds = np.array(new['1d']['indices'])
        # If all or no data selected, wipe histograms
        if len(inds) == 0 or len(inds) == self.numc:
            for key in self.histkeys:
                h = getattr(self,'h_{0}'.format(key))
                h.glyph.top = 'zeros'
        # Otherwise, generate new histogram as appropriate
        else:
            for key in self.histkeys:
                # Recover histogram information
                h = getattr(self,'h_{0}'.format(key))
                arr = getattr(self,'arr_{0}'.format(key))
                edges = getattr(self,'edges_{0}'.format(key))
                # Make new histogram
                hist = (np.histogram(arr[inds],bins=edges)[0]).astype('float')
                # Update glyps
                h.data_source.data['selected'] = hist
                h.glyph.top = 'selected'

    def updateaxlim(self):
        """
        Updates limits of axes and one-to-one line. Used in several callbacks.
        """
        # Find out what data is currently being used
        xkey = self.labels[self.xradio.active]
        ykey = self.labels[self.yradio.active]
        # Find data extremes
        axlims,lineparams = findextremes(self.source.data[xkey],
                                         self.source.data[ykey],
                                         pad=self.pad)
        xmin,xmax,ymin,ymax=axlims
        minlim,maxlim = lineparams

        # Use fixed limits if metrics have it
        if xkey == 'Efficiency' or xkey == 'Completeness':
            xmin = -0.1
            xmax = 1.1
            minlim = np.min([xmin,minlim])
            maxlim = np.min([xmax,maxlim])

        if xkey == 'Found Silhouette' or xkey == 'Matched Silhouette':
            xmin = -1.1
            xmax = 1.1
            minlim = np.min([xmin,minlim])
            maxlim = np.min([xmax,maxlim])

        if ykey == 'Efficiency' or ykey == 'Completeness':
            ymin = -0.1
            ymax = 1.1
            minlim = np.min([ymin,minlim])
            maxlim = np.min([ymax,maxlim])

        if ykey == 'Found Silhouette' or ykey == 'Matched Silhouette':
            ymin = -1.1
            ymax = 1.1
            minlim = np.min([ymin,minlim])
            maxlim = np.min([xmax,maxlim])

        # Find appropriate log scale limits

        if minlim <= 0:
            lminlim = lzp
        else:
            lminlim = minlim

        if xmin <= 0:
            slxmin = zp[self.labels[self.xradio.active]]
        else:
            slxmin = xmin

        if ymin <= 0:
            slymin = zp[self.labels[self.yradio.active]]
        else:
            slymin = ymin

        # Update limits for each panel

        self.p1.x_range.start=xmin
        self.p1.x_range.end=xmax
        self.p1.y_range.start=ymin
        self.p1.y_range.end=ymax

        self.p2.x_range.start = slxmin
        self.p2.x_range.end = xmax
        self.p2.y_range.start = ymin
        self.p2.y_range.end = ymax

        self.p3.x_range.start = xmin
        self.p3.x_range.end = xmax
        self.p3.y_range.start = slymin
        self.p3.y_range.end = ymax

        self.p4.x_range.start = slxmin
        self.p4.x_range.end = xmax
        self.p4.y_range.start = slymin
        self.p4.y_range.end = ymax

        # Update upper limits on histograms
        self.p_found_size.x_range.end = 10**self.maxsize
        self.p_found_size.y_range.end = 1.1*np.max([np.max(self.hsource_found_size.data['backhist']),
                                                    np.max(self.hsource_found_size.data['mainhist'])])

        self.p_matched_size.x_range.end = 10**self.maxsize
        self.p_matched_size.y_range.end = 1.1*np.max([np.max(self.hsource_matched_size.data['backhist']),
                                                      np.max(self.hsource_matched_size.data['mainhist'])])
        
        self.p_efficiency.y_range.end = 1.1*np.max(self.hsource_efficiency.data['mainhist'])
        
        self.p_completeness.y_range.end = 1.1*np.max(self.hsource_completeness.data['mainhist'])
        
        self.p_found_silhouette.y_range.end = 1.1*np.max(self.hsource_found_silhouette.data['mainhist'])
        
        self.p_matched_silhouette.y_range.end = 1.1*np.max([np.max(self.hsource_matched_silhouette.data['backhist']),
                                                            np.max(self.hsource_matched_silhouette.data['mainhist'])])


        # Update limits on the one-to-one lines in each panel
        self.l1.data_source.data['x'] = [minlim,maxlim]
        self.l1.data_source.data['y'] = [minlim,maxlim]

        self.l2.data_source.data['x'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)
        self.l2.data_source.data['y'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)

        self.l3.data_source.data['x'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)
        self.l3.data_source.data['y'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)

        self.l4.data_source.data['x'] = [lminlim,maxlim]
        self.l4.data_source.data['y'] = [lminlim,maxlim]

    def updateallx(self,new):
        """
        Update x glyphs in the panels. Callback for xradio.

        new:        bokeh mandated arg, key to data to use
        """
        
        for r in self.rs:
            r.glyph.x = self.labels[new]

        self.updateaxlim()
    
        for p in self.ps:
            p.xaxis.axis_label = self.labels[new]

    def updateally(self,new):
        """
        Update y glyphs in the panels. Callback for yradio

        new:        bokeh mandated arg, key to data to use
        """

        for r in self.rs:
            r.glyph.y = self.labels[new] 
        
        self.updateaxlim()

        for p in self.ps:
            p.yaxis.axis_label = self.labels[new]

    def updateparam(self,attr,old,new):
        """
        Update back end data with new parameters. Callback for paramselect.
        Requires loadbutton press for actual plot update.

        attr:       bokeh mandated arg, does nothing
        old:        bokeh mandated arg, does nothing
        new:        bokeh mandated arg, index that specifies the parameter values to use

        """
        # Change color/text of load button for as long as data is loading
        self.loadbutton.button_type='danger'
        self.loadbutton.label = 'Loading'

        # Extract new parameters to use
        eps,min_sample = [i.split('=')[-1] for i in new.split(', ')]
        eps = float(eps)
        min_sample = int(min_sample)
        self.epsval = eps
        self.minval = min_sample
        
        # Read in data with those parameters and update self.sourcedict['newsource']
        self.read_run_data(eps,min_sample,update=True)
        # Update self.sourcedict with new histogram data
        self.histograms(update=True)
        # Update axes to bounds of new data
        self.updateaxlim()
        # Update loadbutton behaviour with new callback arguments (i.e. self.sourcedict has the new data in it in the 'new...' keys)
        self.loadbutton.callback = CustomJS(args=self.sourcedict,code=self.callbackstr)
        # Change the color/text of the load button to indicate that it's ready
        self.loadbutton.button_type='success'
        self.loadbutton.label = 'Click to load new data'

    def updatedtype(self,attr,old,new):
        """
        Update back end data with new datatype. Callback for selectdtype.
        Requires loadbutton press for actual plot update.

        attr:       bokeh mandated arg, does nothing
        old:        bokeh mandated arg, does nothing
        new:        bokeh mandated arg, index that specifies the datatype to use
        """
        # Change color/text of load button for as long as data is loading
        self.loadbutton.button_type='danger'
        self.loadbutton.label = 'Loading'

        # Extract datatype to use and read corresponding data
        self.dtype = nametypes[new]
        self.read_base_data(datatype=self.dtype)

        # Are there are any parameters that work? If not, report to user
        if self.allbad:
            print("Didn't find any clusters for any parameter choices with {0} this run".format(typenames[self.dtype]))
            self.loadbutton.button_type='warning'
            self.loadbutton.label = 'No new data to load'

        elif not self.allbad:
            # Update parameter dropdown menu for new dtype
            self.selectparam.options = self.paramlist
            # Read parameters
            eps,min_sample = [i.split('=')[-1] for i in self.selectparam.value.split(', ')]
            eps = float(eps)
            min_sample = int(min_sample)
            match = np.where((eps==self.eps) & (min_sample==self.min_samples))[0]
            # If these parameters don't exist for that datatype, use defaults
            if len(match) < 1: 
                self.selectparam.value = self.paramchoices[self.goodind]
                eps,min_sample = [i.split('=')[-1] for i in self.selectparam.value.split(', ')]
                eps = float(eps)
                min_sample = int(min_sample)
            elif len(match) >= 1:
                if len(match) > 1:
                    warning.warn('This file is lacking unique identifiers - multiple runs with the same parameters')
                m = match[0]
                # If parameters exist but not clusters were found, use defaults
                if m not in np.array(self.goodinds):
                    self.selectparam.value = self.paramchoices[self.goodind]
                    eps,min_sample = [i.split('=')[-1] for i in self.selectparam.value.split(', ')]
                    eps = float(eps)
                    min_sample = int(min_sample)

            self.epsval = eps
            self.minval = min_sample
            
            # read in new self.source
            self.read_run_data(eps,min_sample,update=True)
            # Update self.sourcedict with new histogram data
            self.histograms(update=True)
            # Update axes to bounds of new data
            self.updateaxlim()

            # Update loadbutton behaviour with new callback arguments (i.e. self.sourcedict has the new data in it in the 'new...' keys)
            self.loadbutton.callback = CustomJS(args=self.sourcedict,code=self.callbackstr)
            # Change the color/text of the load button to indicate that it's ready
            self.loadbutton.button_type='success'
            self.loadbutton.label = 'Click to load new data'

    def updatecase(self,attr,old,new):
        """
        Changes global case parameter and loads in new possible timestamps, selecting the first for inspection. This change triggers updatetime. This is a callback for selectcase.
        
        attr:       bokeh mandated arg, does nothing
        old:        bokeh mandated arg, does nothing
        new:        bokeh mandated arg, index that specifies the case to use

        """

        # Change color/text of load button for as long as data is loading
        self.loadbutton.button_type='danger'
        self.loadbutton.label = 'Loading'

        # Get new case
        self.case = new

        # While we're here, check if there's any new data generated
        cases = create_case_list()

        # Find all timestamps for this case
        times = create_time_list(self.case)

        # Update available cases and timestamps
        self.selectcase.options = list(cases)
        self.selecttime.options = list(times)
        # Call updatetime
        self.selecttime.value = times[0]

    def updatetime(self,attr,old,new):
        """
        Update back end data with new timestamp. Callback for selecttime.
        Requires loadbutton press for actual plot update.
        
        attr:       bokeh mandated arg, does nothing
        old:        bokeh mandated arg, does nothing
        new:        bokeh mandated arg, index that specifies the time to use
        """
        # Change color/text of load button for as long as data is loading
        self.loadbutton.button_type='danger'
        self.loadbutton.label = 'Loading'

        # Get new timestamp
        self.timestamp = new

        # Find current datatype
        dtype = nametypes[self.selectdtype.value]

        # Read in basic data for this run (true cluster properties)
        self.read_base_data(case=self.case,timestamp=self.timestamp,datatype=dtype)

        # Update dtype options for this file
        self.selectdtype.options = self.alldtypes

        # If there are no parameters that found clusters, report it
        if self.allbad:
            print("Didn't find any clusters for any parameter choices with {0} this run".format(typenames[self.dtype]))
            self.loadbutton.button_type='danger'
            self.loadbutton.label = 'No new data to load - try another file'
        elif not self.allbad:
            # Update parameter options for this
            self.selectparam.options = self.paramlist
            # Read parameters
            eps,min_sample = [i.split('=')[-1] for i in self.selectparam.value.split(', ')]
            eps = float(eps)
            min_sample = int(min_sample)
            match = np.where((eps==self.eps) & (min_sample==self.min_samples))[0]
            # If these parameters don't exist for that datatype, use defaults
            if len(match) < 1: 
                self.selectparam.value = self.paramchoices[self.goodind]
                eps,min_sample = [i.split('=')[-1] for i in self.selectparam.value.split(', ')]
                eps = float(eps)
                min_sample = int(min_sample)
            elif len(match) >= 1:
                if len(match) > 1:
                    warning.warn('This file is lacking unique identifiers - multiple runs with the same parameters')
                m = match[0]
                # If parameters exist but not clusters were found, use defaults
                if m not in np.array(self.goodinds):
                    self.selectparam.value = self.paramchoices[self.goodind]
                    eps,min_sample = [i.split('=')[-1] for i in self.selectparam.value.split(', ')]
                    eps = float(eps)
                    min_sample = int(min_sample)
            
            self.epsval = eps
            self.minval = min_sample

            # read in new self.source
            self.read_run_data(eps,min_sample,update=True)
            # Update self.sourcedict with new histogram data
            self.histograms(update=True)
            # Update axes to bounds of new data
            self.updateaxlim()

            # Update loadbutton behaviour with new callback arguments (i.e. self.sourcedict has the new data in it in the 'new...' keys)
            self.loadbutton.callback = CustomJS(args=self.sourcedict,code=self.callbackstr)
            # Change the color/text of the load button to indicate that it's ready
            self.loadbutton.button_type='success'
            self.loadbutton.label = 'Click to load new data'



class display_summary(read_results):

    def __init__(self, datatype = 'spec', case = 7, 
                 timestamp = '2018-06-29.20.20.13.729481'):
        read_results.__init__(self,datatype=datatype,case=case,
                              timestamp=timestamp)
        set_colors(self)
        self.layout_plots()
        
# MODIFY FOR OLD SOURCE
    def JScallback(self):
        """
        Makes custom JavaScript callback from bokeh so you can easily swap source dictionaries.
        """ 

        self.callbackstr =''         

        keylist = list(self.sourcedict.keys())
        keylist.remove('button')
        # Initizlize variables
        for key in list(keylist):
            self.callbackstr += """
var v{0} = {0}.data;""".format(key)

        # Update contents of the variables
        for key in list(keylist):
            if 'new' not in key:
                self.callbackstr += """
for (key in vnew{0}) {{
    v{0}[key] = [];
    for (i=0;i<vnew{0}[key].length;i++){{
    v{0}[key].push(vnew{0}[key][i]);
    }}
}}""".format(key)
    
        # Push changes to the variables
        for key in list(keylist):
            if 'new' not in key:
                self.callbackstr += """
{0}.change.emit();""".format(key)
        self.callbackstr += """
button.label = 'I do nothing until you select new run info above';
button.button_type = 'warning';"""

    def layout_plots(self):
        self.read_base_data()
        if self.allbad:
            print("Didn't find any clusters for any parameter choices with {0} this run".format(typenames[self.dtype]))
        elif not self.allbad:
            self.generate_average_stats()
            self.stat_plots()
            self.buttons()

            buttons = column(widgetbox(self.selectcase,width=320,height=30),
                             widgetbox(self.selecttime,width=320,height=30),
                             widgetbox(self.loadbutton,width=320,height=30),
                             widgetbox(self.minsize,width=320,height=30),
                             self.s5)
            avgplots = row(column(self.s1,
                                  self.s2),
                           column(self.s4,
                                  self.s3))
            topplots = row(buttons,avgplots)

            # Here's where you decide the distribution of plots
            self.layout = topplots

            curdoc().add_root(self.layout)
            curdoc().title = "DBSCAN results"

    def stat_plots(self):
        """
        Create summary statistic plots and the dummy plot to hold their
        legend.

        Returns None
        """

        # Number of clusters found
        self.s1 = figure(plot_width=300,plot_height=250,min_border=10,
                         x_axis_location='below', y_axis_location='left',
                         x_axis_type='linear',y_axis_type='log',
                         toolbar_location=None,
                         y_axis_label='Number Found')
        self.s1.y_range.start = 0.5
        self.s1.background_fill_color = self.bcolor
        for d,dtype in enumerate(self.alldtypes):
            dtype = nametypes[dtype]
            c1 = self.s1.scatter(x='xvals',y='numc',source=getattr(self,'{0}_statsource'.format(dtype)),color=typecolor[dtype],size=5,alpha=0.6)
            setattr(self,'{0}_c1'.format(dtype),c1)
            c1l = self.s1.line(x='xvals',y='tnumc',source=getattr(self,'{0}_statsource'.format(dtype)),color=self.outcolor)
            setattr(self,'{0}_c1l'.format(dtype),c1l)
        self.label_stat_xaxis(self.s1,dtype=self.dtype)

        # Average efficiency
        self.s2 = figure(plot_width=300,plot_height=250,min_border=10,
                         x_axis_location='below', y_axis_location='left',
                         x_axis_type='linear',y_axis_type='linear',
                         toolbar_location=None,
                         y_range=(-0.03,1.03),y_axis_label='Efficiency')
        self.s2.background_fill_color = self.bcolor
        for d,dtype in enumerate(self.alldtypes):
            dtype = nametypes[dtype]
            c2 = self.s2.scatter(x='xvals',y='avgeff',source=getattr(self,'{0}_statsource'.format(dtype)),color=typecolor[dtype],size=5,alpha=0.6)
            setattr(self,'{0}_c2'.format(dtype),c2)
        self.label_stat_xaxis(self.s2,dtype=self.dtype)

        # Average completeness
        self.s3 = figure(plot_width=300,plot_height=250,min_border=10,
                         x_axis_location='below', y_axis_location='left',
                         x_axis_type='linear',y_axis_type='linear',
                         toolbar_location=None,
                         y_range=(-0.03,1.03),y_axis_label='Completeness')
        self.s3.background_fill_color = self.bcolor
        for d,dtype in enumerate(self.alldtypes):
            dtype = nametypes[dtype]
            c3 = self.s3.scatter(x='xvals',y='avgcom',source=getattr(self,'{0}_statsource'.format(dtype)),color=typecolor[dtype],size=5,alpha=0.6)
            setattr(self,'{0}_c3'.format(dtype),c3)
        self.label_stat_xaxis(self.s3,dtype=self.dtype)

        # Average silhouette coefficient
        self.s4 = figure(plot_width=300,plot_height=250,min_border=10,
                         x_axis_location='below', y_axis_location='left',
                         x_axis_type='linear',y_axis_type='linear',
                         toolbar_location=None,
                         y_range=(-1.06,1.06),y_axis_label='Found Silhouette')
        self.s4.background_fill_color = self.bcolor
        for d,dtype in enumerate(self.alldtypes):
            dtype = nametypes[dtype]
            c4 = self.s4.scatter(x='xvals',y='avgfsi',source=getattr(self,'{0}_statsource'.format(dtype)),color=typecolor[dtype],size=5,alpha=0.6)
            setattr(self,'{0}_c4'.format(dtype),c4)
        self.label_stat_xaxis(self.s4,dtype=self.dtype)

        # Dummy plot to generate the legend
        items = []
        self.s5 = figure(plot_width=200,plot_height=200,
                         x_axis_location=None,y_axis_location=None,
                         toolbar_location=None,x_range=(0,1),y_range=(0,1))
        # Make it invisible
        self.s5.background_fill_color = None
        self.s5.xgrid.visible = False
        self.s5.ygrid.visible = False
        self.s5.outline_line_color = None
        for d,dtype in enumerate(list(nametypes.keys())):
            dtype = nametypes[dtype]
            c5 = self.s5.scatter(x=[0.5],y=[0.5],color=typecolor[dtype],size=5,alpha=0.6)
            items.append((typenames[dtype],[c5]))
        legend = Legend(items=items, location=(0,30))
        self.s5.add_layout(legend, 'center')
        self.s5.legend.background_fill_alpha = 1


    def label_stat_xaxis(self,plot,dtype='spec'):
        """
        Labels the x-axis of a plot with the parameter values instead of dummy index.

        plot:       figure object with x-axis to relabel
        """
        ss = getattr(self,'{0}_statsource'.format(dtype))
        xvals = list(ss.data['xvals'])
        param = list(ss.data['params'])
        plot.xaxis.ticker = xvals
        overrides = dict(zip(list(np.array(xvals).astype(str)), param))
        plot.xaxis.major_label_overrides = overrides 
        plot.xaxis.major_label_orientation = np.pi/4

    def buttons(self):
        """
        Creates buttons that manipulate the plots
        """

        cases = create_case_list()
        times = create_time_list(self.case)

        # Creates text input to choose minimum cluster size to consider
        self.minsize = TextInput(value="1", title="Minimum size - choose between 1 and {0}:".format(int(self.maxmem)))
        self.minsize.on_change('value',self.updatestatplot)


# Create drop down menu for possible cases
        self.selectcase = Select(title='case',value=self.case,options=list(cases))
        self.selectcase.on_change('value',self.updatecase)

        # Create drop down menu for possible timestamps
        self.selecttime = Select(title='timestamp',value=self.timestamp,options=list(times))
        self.selecttime.on_change('value',self.updatetime)

        # Create button to actually push results of new data to plot
        self.loadbutton = Button(label='I do nothing until you select new run info above', button_type='warning')
        self.sourcedict['button'] = self.loadbutton
        self.JScallback()
        self.loadbutton.callback = CustomJS(args=self.sourcedict,code=self.callbackstr)

    def updatecase(self,attr,old,new):
        """
        Changes global case parameter and loads in new possible timestamps, selecting the first for inspection. This change triggers updatetime. This is a callback for selectcase.
        
        attr:       bokeh mandated arg, does nothing
        old:        bokeh mandated arg, does nothing
        new:        bokeh mandated arg, index that specifies the case to use

        """

        # Change color/text of load button for as long as data is loading
        self.loadbutton.button_type='danger'
        self.loadbutton.label = 'Loading'

        # Get new case
        self.case = new

        # While we're here, check if there's any new data generated
        cases = create_case_list()

        # Find all timestamps for this case
        times = create_time_list(self.case)

        # Update available cases and timestamps
        self.selectcase.options = list(cases)
        self.selecttime.options = list(times)
        # Call updatetime
        self.selecttime.value = times[0]

    def updatetime(self,attr,old,new):
        """
        Update back end data with new timestamp. Callback for selecttime.
        Requires loadbutton press for actual plot update.
        
        attr:       bokeh mandated arg, does nothing
        old:        bokeh mandated arg, does nothing
        new:        bokeh mandated arg, index that specifies the time to use
        """
        # Change color/text of load button for as long as data is loading
        self.loadbutton.button_type='danger'
        self.loadbutton.label = 'Loading'

        # Get new timestamp
        self.timestamp = new

        # Read in basic data for this run (true cluster properties)
        self.read_base_data(case=self.case,timestamp=self.timestamp)

        # If there are no parameters that found clusters, report it
        if self.allbad:
            print("Didn't find any clusters for any parameter choices with {0} this run".format(typenames[self.dtype]))
            self.loadbutton.button_type='danger'
            self.loadbutton.label = 'No new data to load - try another file'
        elif not self.allbad:
            # Get new average stats for this run
            self.generate_average_stats(minmem=int(self.minsize.value),update=True)
            # Update loadbutton behaviour with new callback arguments (i.e. self.sourcedict has the new data in it in the 'new...' keys)
            self.loadbutton.callback = CustomJS(args=self.sourcedict,code=self.callbackstr)
            # Change the color/text of the load button to indicate that it's ready
            self.loadbutton.button_type='success'
            self.loadbutton.label = 'Click to load new data'

    def updatestatplot(self, attr, old, new):
        """
        Update summary plots with a new choice for the minimum number of members

        attr:       bokeh mandated arg, does nothing
        old:        bokeh mandated arg, does nothing
        new:        bokeh mandated arg, specifies minimum members
        """
        # Find the minimum
        num = int(new)
        # Recalculate averages - includes natural zeroing if datatype not present
        self.generate_average_stats(minmem=num)
        # Cycle through available data types  and update glyphs
        for dtype in self.alldtypes:
            dtype = nametypes[dtype]
            # extract each plot for this datatype
            ss = getattr(self,'{0}_statsource'.format(dtype))
            c1 = getattr(self,'{0}_c1'.format(dtype))
            c1l = getattr(self,'{0}_c1l'.format(dtype))
            c2 = getattr(self,'{0}_c2'.format(dtype))
            c3 = getattr(self,'{0}_c3'.format(dtype))
            c4 = getattr(self,'{0}_c4'.format(dtype))

            # Change source and update glyphs

            # Number of clusters found
            c1.data_source.data = ss.data
            c1.glyph.y = 'numc'
            # Total number of clusters put in
            c1l.data_source.data = ss.data
            c1l.glyph.y = 'tnumc'
            # Average efficiency
            c2.data_source.data = ss.data
            c2.glyph.y = 'avgeff'
            # Average completeness
            c3.data_source.data = ss.data
            c3.glyph.y = 'avgcom'
            # Average silhouette coefficient
            c4.data_source.data = ss.data
            c4.glyph.y = 'avgfsi'

if __name__ == '__main__':
    singlerun = display_single(case='8',timestamp='2018-07-25.12.13.55.213653',datatype='spec',pad=0.1)
    summaryrun = display_summary(case='8',timestamp='2018-07-25.12.13.55.213653',datatype='spec')
