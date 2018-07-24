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

inkscape = os.getenv('INKSCAPE',False)
if not inkscape:
    warnings.warn('Need Inkscape to convert saved svg to pdf')

# Types of data that clustering was run on

typenames = {'spec':'spectra','abun':'abundances','toph':'tophat windows','wind':'windows'}
nametypes = {'spectra':'spec','abundances':'abun','tophat windows':'toph','windows':'wind'}

zp = {'Efficiency':5e-3,'Completeness':5e-3,'Found Silhouette':5e-3,'Matched Silhouette':5e-3,'Found Size':0.5,'Matched Size':0.5}
lzp=1e-3
plot_eps = 0.5
padfac = 0.1

def findextremes(x,y,pad=0.1):
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    xmin -= padfac*pad*xmax
    xmax += pad*xmax
    ymin -= padfac*pad*ymax
    ymax += pad*ymax
    minlim = np.min([xmin,ymin])
    maxlim = np.max([xmax,ymax])
    return (xmin,xmax,ymin,ymax),[minlim,maxlim]

class read_results(object):

    def __init__(self,ind = 0, datatype = 'spec', case = 7, 
                 timestamp = '2018-07-06.14.28.11.782906'):
        self.ind = ind
        self.dtype = datatype
        self.case = case
        self.timestamp = timestamp

    def read_base_data(self,datatype=None,case=None,timestamp=None,neighbours=20):
        self.allbad = False
        self.data = h5py.File('case{0}_{1}.hdf5'.format(self.case,self.timestamp),'r+')
        if case:
            self.case = case
        if timestamp:
            self.timestamp = timestamp
        alldtypes = np.unique(np.array([i.split('_')[0] for i in list(self.data.keys())]))
        special = np.where((alldtypes=='true') | (alldtypes=='labels'))
        if len(special[0])>0:
            alldtypes = np.delete(alldtypes,special[0])
        self.alldtypes = []
        for dtype in alldtypes:
            self.alldtypes.append(typenames[dtype])
        #self.chemspace = self.data['{0}'.format(self.dtype)][:]
        #self.labels_true = self.data['labels_true'][:]
        self.tsize = self.data['true_size'][:]
        self.read_dtype_data(datatype=dtype)
        
    def read_dtype_data(self,datatype='spec'):
        if datatype:
            self.dtype = datatype
        self.tsil = self.data['{0}_true_sil_neigh{1}'.format(self.dtype,neighbours)][:]
        # scrub nans
        self.tsil[np.isnan(self.tsil)]=-1
        self.labels_pred = self.data['{0}_labels_pred'.format(self.dtype)][:]
        self.numcs = []
        self.numms = []
        self.goodind = 0
        for row in range(self.labels_pred.shape[0]):
            labcount,labs = membercount(self.labels_pred[row])
            bad = np.where(labs==-1)
            if len(bad[0])>0:
                labs = np.delete(labs,bad[0][0])
                labcount = np.delete(labcount,bad[0][0])
            self.numcs.append(len(labs))
            self.numms.append(labcount)
        self.numcs = np.array(self.numcs)
        self.goodinds = np.where(self.numcs > 3)
        if len(self.goodinds[0]) > 0:
            self.goodind = self.goodinds[0][0]
        elif len(self.goodinds[0])==0:
            self.allbad = True
        self.min_samples = self.data.attrs['{0}_min'.format(self.dtype)][:]
        self.eps = self.data.attrs['{0}_eps'.format(self.dtype)][:]
        self.epsval = self.eps[self.goodind]
        self.minval = self.min_samples[self.goodind]
        self.paramchoices = []
        self.ticklabels = []
        for i in range(len(self.eps)):
            self.paramchoices.append('eps={0}, min={1}'.format(self.eps[i],self.min_samples[i]))
            self.ticklabels.append('{0}, {1}'.format(self.eps[i],self.min_samples[i]))
        self.paramlist = list(np.array(self.paramchoices)[self.goodinds])

    def generate_average_stats(self,minmem=1,update=False):
        vintdtype = copy.deepcopy(self.dtype)
        self.maxmem = 1
        labmaster = []

        for d,dtype in enumerate(self.alldtypes):
            dtype = nametypes[dtype]
            self.read_dtype_data(datatype=dtype)
            labmaster.append(self.ticklabels)

        labmaster = np.unique(np.array([item for sublist in labmaster for item in sublist]))
        xvals = np.arange(len(labmaster))

        for d,dtype in enumerate(self.alldtypes):
            dtype = nametypes[dtype]
            self.read_dtype_data(datatype=dtype)
            effs = np.zeros(len(labmaster))
            coms = np.zeros(len(labmaster))
            fsil = -np.ones(len(labmaster))
            msil = -np.ones(len(labmaster))
            numc = 0.01*np.ones(len(labmaster))
            alph = 0.7*np.ones(len(labmaster))
            for e,eps in enumerate(self.eps):
                if e in self.goodinds[0]:
                    sizes = self.numms[e]
                    try:
                        self.read_run_data(eps=eps,min_sample=self.min_samples[e],update=True)
                        vals = sizes > minmem
                        if np.sum(vals) > 0:
                            match = np.where(labmaster=='{0}, {1}'.format(eps,self.min_samples[e]))
                            numc[match] = len(sizes[vals])
                            effs[match] = np.mean(self.eff[vals])
                            coms[match] = np.mean(self.com[vals])
                            fsil[match] = np.mean(self.fsil[vals])
                            msil[match] = np.mean(self.msil[vals])
                        self.maxmem = np.max([self.maxmem,np.max(sizes)])
                    except:
                        pass
            tnumc = np.array([len(self.tsize[self.tsize>minmem])]*len(labmaster))
            tnumc[tnumc < 1] = 0.01
            statsource = {'params':labmaster,'numc':numc,
                               'avgeff':effs,'avgcom':coms,
                               'avgfsi':fsil,'avgmsi':msil,
                               'xvals':xvals,'alphas':alph,
                               'tnumc':tnumc}
            setattr(self,'{0}_statsource'.format(dtype),ColumnDataSource(statsource))
            if not update:
                self.sourcedict['{0}source'.format(dtype)] = getattr(self,'{0}_statsource'.format(dtype))
            self.sourcedict['new{0}source'.format(dtype)] = getattr(self,'{0}_statsource'.format(dtype))
        self.dtype = vintdtype

    def read_run_data(self,eps=None,min_sample=None,update=False):
        if eps:
            self.epsval = eps
        if min_sample:
            self.minval = min_sample
        self.matchtlabs = self.data['{0}_match_tlabs_eps{1}_min{2}'.format(self.dtype,self.epsval,self.minval)][:]
        if len(self.matchtlabs) > 0:
            self.msil = self.tsil[self.matchtlabs]
            self.msize = self.tsize[self.matchtlabs]
        elif len(self.matchtlabs) == 0:
            self.msil = np.array([])
            self.msize = np.array([])
        self.fsil = self.data['{0}_found_sil_eps{1}_min{2}_neigh{3}'.format(self.dtype,self.epsval,self.minval,neighbours)][:]
        self.eff = self.data['{0}_eff_eps{1}_min{2}'.format(self.dtype,self.epsval,self.minval)][:]
        self.com = self.data['{0}_com_eps{1}_min{2}'.format(self.dtype,self.epsval,self.minval)][:]
        self.fsize = self.data['{0}_found_size_eps{1}_min{2}'.format(self.dtype,self.epsval,self.minval)][:]
        self.numc = len(self.fsize)
        # Scrub nans
        self.msil[np.isnan(self.msil)] = -1
        self.fsil[np.isnan(self.fsil)] = -1
        self.eff[np.isnan(self.eff)] = 0
        self.com[np.isnan(self.com)] = 0
        self.datadict = {'Efficiency':self.eff,'Completeness':self.com,
                         'Found Silhouette':self.fsil,'Matched Silhouette':self.msil,
                         'Found Size':self.fsize,'Matched Size':self.msize}
        self.source=ColumnDataSource(data=self.datadict)
        if not update:
            self.sourcedict = {'source':self.source,'newsource':self.source}

class display_result(read_results):

    def __init__(self,ind = 0, datatype = 'spec', case = 7, 
                 timestamp = '2018-06-29.20.20.13.729481', pad = 0.1, 
                 backgroundhist = True,sqside = 460,tools=TOOLS):
        read_results.__init__(self,ind=ind,datatype=datatype,case=case,
                              timestamp=timestamp)
        self.sqside=sqside
        self.backgroundhist=backgroundhist
        self.tools=tools
        self.pad = pad
        self.set_colors()
        self.layout_plots()
        
# MODIFY FOR OLD SOURCE

    def JScallback(self):
        """
        Makes custom JavaScript callback from bokeh so you can easily swap source dictionaries.
        """ 

        self.callbackstr =''         

        for key in list(self.sourcedict.keys()):
            self.callbackstr += """
var v{0} = {0}.data;""".format(key)

        for key in list(self.sourcedict.keys()):
            if 'new' not in key:
                self.callbackstr += """
for (key in vnew{0}) {{
    v{0}[key] = [];
    for (i=0;i<vnew{0}[key].length;i++){{
    v{0}[key].push(vnew{0}[key][i]);
    }}
}}""".format(key)
    
        for key in list(self.sourcedict.keys()):
            if 'new' not in key:
                self.callbackstr += """
{0}.change.emit();""".format(key)

    def layout_plots(self):
        self.read_base_data()
        if self.allbad:
            print("Didn't find any clusters for any parameter choices with {0} this run".format(typenames[self.dtype]))
        elif not self.allbad:
            self.read_run_data()
            self.generate_average_stats()
            self.stat_plots()
            self.center_plot()
            self.histograms()
            self.buttons()

            buttons = column(widgetbox(self.minsize,width=200,height=30),
                             widgetbox(self.selectcase,width=200,height=30),
                             widgetbox(self.selecttime,width=200,height=30),
                             widgetbox(self.selectdtype,width=200,height=30),
                             widgetbox(self.selectparam,width=200,height=30),
                             widgetbox(self.loadbutton,width=200,height=30))
            avgplots = row(column(self.s1,
                                  self.s2),
                           column(self.s4,
                                  self.s3))
            topplots = row(buttons,avgplots)

            mainplot = row(column(Tabs(tabs=self.panels,width=self.sqside+20)),
                           column(Spacer(width=440,height=50),
                                  row(Spacer(width=50,height=30),
                                      widgetbox(self.toggleline,width=300)),
                                  row(Spacer(width=50,height=100),
                                      widgetbox(self.xradio,width=440)),
                                  row(Spacer(width=50,height=100),
                                      widgetbox(self.yradio,width=440)),
                                  row(self.p_found_size,
                                      self.p_matched_size)))

            histplot = row(self.p_efficiency,
                           self.p_completeness,
                           self.p_found_silhouette,
                           self.p_matched_silhouette)
            # Here's where you decide the distribution of plots
            self.layout = column(topplots,mainplot,histplot)

            # Activate buttons
            self.r1.data_source.on_change('selected', self.updatetophist)
            self.r2.data_source.on_change('selected', self.updatetophist)
            self.r3.data_source.on_change('selected', self.updatetophist)
            self.r4.data_source.on_change('selected', self.updatetophist)
            self.xradio.on_click(self.updateallx)
            self.yradio.on_click(self.updateally)
            self.p1.on_event(events.Reset,self.resetplots)
            self.p2.on_event(events.Reset,self.resetplots)
            self.p3.on_event(events.Reset,self.resetplots)
            self.p4.on_event(events.Reset,self.resetplots)
            # self.p1.on_event(events.MouseEnter,self.updateparam)
            # self.p2.on_event(events.MouseEnter,self.updateparam)
            # self.p3.on_event(events.MouseEnter,self.updateparam)
            # self.p4.on_event(events.MouseEnter,self.updateparam)
            curdoc().add_root(self.layout)
            curdoc().title = "DBSCAN on {0} with eps={1}, min_samples={2}".format(typenames[self.dtype], 
                                                                                  self.epsval,self.minval)

    def set_colors(self):
        self.bcolor = "#FFF7EA" #cream
        self.unscolor = "#4C230A" #dark brown
        self.maincolor = "#A53F2B" #dark red
        self.histcolor = "#F6BD60" #yellow
        self.outcolor = "#280004" #dark red black
        self.color1 = "#F98D20" # light orange
        self.color2 = "#904C77" # purple
        self.color3 = "#79ADDC" # light blue
        self.color4 = "#ED6A5A" # orange
        self.colorlist = [self.color1,self.color2,self.color3,self.color4]
        return [self.bcolor,self.unscolor,self.outcolor,
                self.histcolor,self.maincolor]

    def stat_plots(self):
        self.s1 = figure(plot_width=300,plot_height=250,min_border=10,
                         x_axis_location='below', y_axis_location='left',
                         x_axis_type='linear',y_axis_type='log',
                         output_backend='svg',toolbar_location=None,
                         y_axis_label='Number Found')
        self.s1.y_range.start = 1
        self.s1.background_fill_color = self.bcolor
        for d,dtype in enumerate(self.alldtypes):
            dtype = nametypes[dtype]
            c1 = self.s1.scatter(x='xvals',y='numc',source=getattr(self,'{0}_statsource'.format(dtype)),color=self.colorlist[d],size=5,alpha=0.6)
            setattr(self,'{0}_c1'.format(dtype),c1)
            c1l = self.s1.line(x='xvals',y='tnumc',source=getattr(self,'{0}_statsource'.format(dtype)),color=self.colorlist[d])
            setattr(self,'{0}_c1l'.format(dtype),c1l)
        self.label_stat_xaxis(self.s1,dtype=self.dtype)

        self.s2 = figure(plot_width=300,plot_height=250,min_border=10,
                         x_axis_location='below', y_axis_location='left',
                         x_axis_type='linear',y_axis_type='linear',
                         output_backend='svg',toolbar_location=None,
                         y_range=(-0.03,1.03),y_axis_label='Efficiency')
        self.s2.background_fill_color = self.bcolor
        items = []
        for d,dtype in enumerate(self.alldtypes):
            dtype = nametypes[dtype]
            c2 = self.s2.scatter(x='xvals',y='avgeff',source=getattr(self,'{0}_statsource'.format(dtype)),color=self.colorlist[d],size=5,alpha=0.6)
            setattr(self,'{0}_c2'.format(dtype),c2)
            items.append((typenames[dtype],[getattr(self,'{0}_c2'.format(dtype))]))
        self.label_stat_xaxis(self.s2,dtype=self.dtype)

        self.s3 = figure(plot_width=300,plot_height=250,min_border=10,
                         x_axis_location='below', y_axis_location='left',
                         x_axis_type='linear',y_axis_type='linear',
                         output_backend='svg',toolbar_location=None,
                         y_range=(-0.03,1.03),y_axis_label='Completeness')
        self.s3.background_fill_color = self.bcolor
        for d,dtype in enumerate(self.alldtypes):
            dtype = nametypes[dtype]
            c3 = self.s3.scatter(x='xvals',y='avgcom',source=getattr(self,'{0}_statsource'.format(dtype)),color=self.colorlist[d],size=5,alpha=0.6)
            setattr(self,'{0}_c3'.format(dtype),c3)
        self.label_stat_xaxis(self.s3,dtype=self.dtype)

        self.s4 = figure(plot_width=300,plot_height=250,min_border=10,
                         x_axis_location='below', y_axis_location='left',
                         x_axis_type='linear',y_axis_type='linear',
                         output_backend='svg',toolbar_location=None,
                         y_range=(-1.06,1.06),y_axis_label='Found Silhouette')
        self.s4.background_fill_color = self.bcolor
        
        for d,dtype in enumerate(self.alldtypes):
            dtype = nametypes[dtype]
            c4 = self.s4.scatter(x='xvals',y='avgfsi',source=getattr(self,'{0}_statsource'.format(dtype)),color=self.colorlist[d],size=5,alpha=0.6)
            setattr(self,'{0}_c4'.format(dtype),c4)
        self.label_stat_xaxis(self.s4,dtype=self.dtype)

        legend = Legend(items=items, location=(0,30))

        self.s2.add_layout(legend, 'right')


    def label_stat_xaxis(self,plot,dtype='sepc'):
        ss = getattr(self,'{0}_statsource'.format(dtype))
        xvals = list(ss.data['xvals'])
        param = list(ss.data['params'])
        plot.xaxis.ticker = xvals
        overrides = dict(zip(list(np.array(xvals).astype(str)), param))
        plot.xaxis.major_label_overrides = overrides 
        plot.xaxis.major_label_orientation = np.pi/4

    def center_plot(self,xlabel=None,ylabel=None):
        self.panels = []
        if not xlabel:
            xlabel = 'Efficiency'
        if not ylabel:
            ylabel = 'Completeness'
        x = self.source.data[xlabel]
        y = self.source.data[ylabel]
        axlims,lineparams = findextremes(x,y,pad=self.pad)
        xmin,xmax,ymin,ymax=axlims
        minlim,maxlim = lineparams

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
                         x_range=(xmin,xmax),y_range=(ymin,ymax),
                         output_backend="svg")
        self.p1.background_fill_color = self.bcolor
        self.p1.select(BoxSelectTool).select_every_mousemove = False
        self.p1.select(LassoSelectTool).select_every_mousemove = False

        self.r1 = self.p1.scatter(x=xlabel, y=ylabel, source=self.source, 
                                  size=3, color=self.maincolor, alpha=0.6)
        self.l1 = self.p1.line(lineparams,lineparams,
                               color=self.outcolor)
        self.r1.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        self.panels.append(Panel(child=self.p1,title='linear'))


        # SEMILOGX TAB
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
                         x_range=(slxmin,xmax),y_range=(ymin,ymax),
                         output_backend="svg")
        self.p2.background_fill_color = self.bcolor
        self.p2.select(BoxSelectTool).select_every_mousemove = False
        self.p2.select(LassoSelectTool).select_every_mousemove = False

        self.r2 = self.p2.scatter(x=xlabel,y=ylabel, source=self.source, 
                                  size=3, color=self.maincolor, alpha=0.6)
    
        self.l2 = self.p2.line(np.logspace(np.log10(lminlim),
                                           np.log10(maxlim),100),
                               np.logspace(np.log10(lminlim),
                                           np.log10(maxlim),100),
                               color=self.outcolor)
        self.r2.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        self.panels.append(Panel(child=self.p2,title='semilogx'))


        # SEMILOGY TAB
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
                         x_range=(xmin,xmax), y_range=(slymin,ymax),
                         output_backend="svg")
        self.p3.background_fill_color = self.bcolor
        self.p3.select(BoxSelectTool).select_every_mousemove = False
        self.p3.select(LassoSelectTool).select_every_mousemove = False

        self.r3 = self.p3.scatter(x=xlabel,y=ylabel, source=self.source, 
                                  size=3, color=self.maincolor, alpha=0.6)
  
        self.l3 = self.p3.line(np.logspace(np.log10(lminlim),
                                           np.log10(maxlim),100),
                               np.logspace(np.log10(lminlim),
                                           np.log10(maxlim),100),
                               color=self.outcolor)
        self.r3.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        self.panels.append(Panel(child=self.p3,title='semilogy'))


        # LOG PLOT
        self.p4 = figure(tools=TOOLS, plot_width=self.sqside, 
                         plot_height=self.sqside, min_border=10, 
                         min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left', 
                         x_axis_label=xlabel, y_axis_label=ylabel, 
                         x_axis_type='log', y_axis_type='log', 
                         x_range=(slxmin,xmax), y_range=(slymin,ymax),
                         output_backend="svg")
        self.p4.background_fill_color = self.bcolor
        self.p4.select(BoxSelectTool).select_every_mousemove = False
        self.p4.select(LassoSelectTool).select_every_mousemove = False

        self.r4 = self.p4.scatter(x=xlabel,y=ylabel, source=self.source, 
                                  size=3, color=self.maincolor, alpha=0.6)
        self.l4 = self.p4.line([lminlim,maxlim],[lminlim,maxlim],
                               color=self.outcolor)
        self.r4.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        self.panels.append(Panel(child=self.p4,title='log'))

        self.ps = [self.p1,self.p2,self.p3,self.p4]
        self.rs = [self.r1,self.r2,self.r3,self.r4]
        self.ls = [self.l1,self.l2,self.l3,self.l4]

        # NEEDS CHECK FOR <= 0 POINTS

    def make_hist(self,key,bins=20,x_range=(),xscale='linear',
                  yscale='linear',background=[], update=False):
        hist_name = key.lower().replace(' ','_')
        arr = self.source.data[key] 
        setattr(self,'arr_{0}'.format(hist_name),arr)
        hist, edges = np.histogram(arr, bins=bins)
        hist = hist.astype('float')
        setattr(self,'hist_{0}'.format(hist_name),hist)
        setattr(self,'edges_{0}'.format(hist_name),edges)
        zeros = np.zeros(len(edges)-1)
        setattr(self,'zeros_{0}'.format(hist_name),zeros)
        hist_max = max(hist)*1.1
        setattr(self,'hist_max_{0}'.format(hist_name),hist_max)

        if background != []:
            backhist = np.histogram(background,bins=edges)[0]
            backhist = backhist.astype('float')
            setattr(self,'bhist_{0}'.format(hist_name),backhist)
            backmax = np.max(backhist)*1.1
            if backmax > hist_max:
                hist_max = backmax
                setattr(self,'hist_max_{0}'.format(hist_name),hist_max)
        ymax = hist_max
        if x_range==():
            x_range = (np.min(edges),np.max(edges))
        if yscale=='linear':
            ymin = 0
        elif yscale=='log':
            ymin=plot_eps+plot_eps/2.
            hist[hist < plot_eps] = plot_eps
            setattr(self,'hist_{0}'.format(hist_name),hist)
            if background != []:
                backhist[backhist < plot_eps] = plot_eps
                setattr(self,'bhist_{0}'.format(hist_name),backhist)
        setattr(self,'ymin_{0}'.format(hist_name),ymin)
        histsource = {'mainhist':hist,'left':edges[:-1],'right':edges[1:],
                      'bottom':ymin*np.ones(len(hist)),'zeros':zeros,'selected':zeros}
        if background != []:
            histsource['backhist'] = backhist
        histsource = ColumnDataSource(data=histsource)
        setattr(self,'hsource_{0}'.format(hist_name),histsource)

        if not update:
            p = figure(toolbar_location=None, plot_width=220, plot_height=200,
                        x_range=x_range,y_range=(ymin, hist_max), min_border=10, 
                        min_border_left=50, y_axis_location="right",
                        x_axis_label=key,x_axis_type=xscale,
                        y_axis_type=yscale,output_backend="svg")
            setattr(self,'p_{0}'.format(hist_name),p)
            p.xgrid.grid_line_color = None
            #pt.yaxis.major_label_orientation = np.pi/4
            p.background_fill_color = self.bcolor

            if background != []:
                bghist = p.quad(bottom='bottom', left='left', right='right', 
                                top='backhist', 
                                source = getattr(self,'hsource_{0}'.format(hist_name)), 
                                color=self.unscolor, line_color=self.outcolor)
            mnhist = p.quad(bottom='bottom', left='left', right='right', 
                         top='mainhist', alpha=0.7,
                         source = getattr(self,'hsource_{0}'.format(hist_name)),
                         color=self.histcolor, line_color=self.outcolor)
            h1 = p.quad(bottom='bottom', left='left', right='right', 
                         top='selected', alpha=0.6, 
                         source = getattr(self,'hsource_{0}'.format(hist_name)),
                         color=self.maincolor,line_color=None)
            setattr(self,'h_{0}'.format(hist_name),h1)


    def histograms(self,nbins=20,update=False):
        self.make_hist('Efficiency',bins=np.linspace(0,1,nbins),
                  x_range=(0,1),yscale='log',update=update)
        self.make_hist('Completeness',bins=np.linspace(0,1,nbins),
                  x_range=(0,1),yscale='log',update=update)
        self.make_hist('Found Silhouette',bins=np.linspace(-1,1,nbins),
                  x_range=(-1,1),yscale='log',update=update)
        self.make_hist('Matched Silhouette',bins=np.linspace(-1,1,nbins),
                  x_range=(-1,1),yscale='log',background=self.tsil,
                  update=update)
        self.maxsize = np.max(np.array([np.max(self.source.data['Found Size']),
                                        np.max(self.source.data['Matched Size']),
                                        np.max(self.tsize)]))
        self.maxsize = np.log10(self.maxsize)
        self.make_hist('Found Size',bins=np.logspace(0,self.maxsize,nbins),
                  xscale='log',yscale='log',background=self.tsize,
                  update=update)
        self.make_hist('Matched Size',bins=np.logspace(0,self.maxsize,nbins),
                  xscale='log',yscale='log',background=self.tsize,
                  update=update)
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
        self.sourcedict['newheff'] = self.hsource_efficiency
        self.sourcedict['newhcom'] = self.hsource_completeness
        self.sourcedict['newhfsi'] = self.hsource_found_silhouette
        self.sourcedict['newhmsi'] = self.hsource_matched_silhouette
        self.sourcedict['newhfsz'] = self.hsource_found_size
        self.sourcedict['newhmsz'] = self.hsource_matched_size



    def buttons(self):

        caselist = glob.glob('*.hdf5')
        cases = np.unique(np.array([i.split('_')[0].split('case')[-1] for i in caselist])).astype('int')
        cases.sort()
        cases = cases.astype('str')

        timelist = glob.glob('case{0}*.hdf5'.format(self.case))
        times = np.array([i.split('_')[1].split('.hdf5')[0] for i in timelist])[::-1]


        self.minsize = TextInput(value="1", title="Minimum size - choose between 1 and {0}:".format(int(self.maxmem)))
        self.minsize.on_change('value',self.updatestatplot)

        self.labels = list(self.source.data.keys())

        self.xradio = RadioButtonGroup(labels=self.labels, active=0,name='x-axis')
        self.yradio = RadioButtonGroup(labels=self.labels, active=1,name='y-axis')

        self.selectcase = Select(title='case',value=self.case,options=list(cases))
        self.selectcase.on_change('value',self.updatecase)
        self.selecttime = Select(title='timestamp',value=self.timestamp,options=list(times))
        self.selecttime.on_change('value',self.updatetime)

        self.selectdtype = Select(title='data type',value='spectra',options=self.alldtypes)
        self.selectdtype.on_change('value',self.updatedtype)
        
        self.selectparam = Select(title="parameter values", value=self.paramchoices[self.goodind], 
                           options=self.paramlist)
        self.selectparam.on_change('value',self.updateparam)
        self.loadbutton = Button(label='Select new run info above', button_type='success')
        self.JScallback()
        self.loadbutton.callback = CustomJS(args=self.sourcedict,code=self.callbackstr)

        self.saveplots = Button(label='Save plots',button_type='primary')
        self.saveplots.on_click(self.exportplots)
        
        code = '''\
        object1.visible = toggle.active
        object2.visible = toggle.active
        object3.visible = toggle.active
        object4.visible = toggle.active
        '''
        linecb = CustomJS.from_coffeescript(code=code, args={})
        self.toggleline = Toggle(label="One-to-one line", button_type="success", active=True,callback=linecb)
        linecb.args = {'toggle': self.toggleline, 'object1': self.l1, 'object2': self.l2, 'object3': self.l3, 'object4': self.l4}

    def resetplots(self,attrs):
        if 'histkeys' not in dir(self):
            histkeys = self.datadict.keys()
            self.histkeys = np.array([key.lower().replace(' ','_') for key in histkeys])
        for key in self.histkeys:
            h = getattr(self,'h_{0}'.format(key))
            h.glyph.top = 'zeros'
        self.updateaxlim()


    def updatetophist(self, attr, old, new):
        if 'histkeys' not in dir(self):
            histkeys = self.datadict.keys()
            self.histkeys = np.array([key.lower().replace(' ','_') for key in histkeys])
        inds = np.array(new['1d']['indices'])
        if len(inds) == 0 or len(inds) == self.numc:
            for key in self.histkeys:
                h = getattr(self,'h_{0}'.format(key))
                h.glyph.top = 'zeros'
        else:
            neg_inds = np.ones_like(self.source.data['Efficiency'], dtype=np.bool)
            neg_inds[inds] = False
            for key in self.histkeys:
                h = getattr(self,'h_{0}'.format(key))
                arr = getattr(self,'arr_{0}'.format(key))
                edges = getattr(self,'edges_{0}'.format(key))
                hist = (np.histogram(arr[inds],bins=edges)[0]).astype('float')
                h.data_source.data['selected'] = hist
                h.glyph.top = 'selected'

    def updateaxlim(self):
        xkey = self.labels[self.xradio.active]
        ykey = self.labels[self.yradio.active]
        axlims,lineparams = findextremes(self.source.data[xkey],
                                         self.source.data[ykey],
                                         pad=self.pad)
        xmin,xmax,ymin,ymax=axlims
        minlim,maxlim = lineparams

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

        self.p_found_size.x_range.end = 10**self.maxsize
        self.p_matched_size.x_range.end = 10**self.maxsize
        self.p_efficiency.y_range.end = 1.1*np.max(self.hsource_efficiency.data['mainhist'])
        self.p_completeness.y_range.end = 1.1*np.max(self.hsource_completeness.data['mainhist'])
        self.p_found_silhouette.y_range.end = 1.1*np.max(self.hsource_found_silhouette.data['mainhist'])


        self.l1.data_source.data['x'] = [minlim,maxlim]
        self.l1.data_source.data['y'] = [minlim,maxlim]

        self.l2.data_source.data['x'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)
        self.l2.data_source.data['y'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)

        self.l3.data_source.data['x'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)
        self.l3.data_source.data['y'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)

        self.l4.data_source.data['x'] = [lminlim,maxlim]
        self.l4.data_source.data['y'] = [lminlim,maxlim]

    def updateallx(self,new):
        
        for r in self.rs:
            r.glyph.x = self.labels[new]

        self.updateaxlim()
    
        for p in self.ps:
            p.xaxis.axis_label = self.labels[new]

    def updateally(self,new):

        for r in self.rs:
            r.glyph.y = self.labels[new] 
        
        self.updateaxlim()

        for p in self.ps:
            p.yaxis.axis_label = self.labels[new]

    def updateparam(self,attr,old,new):
        self.loadbutton.button_type='warning'
        self.loadbutton.label = 'Loading'
        eps,min_sample = [i.split('=')[-1] for i in new.split(', ')]
        eps = float(eps)
        min_sample = int(min_sample)
        # read in new self.source
        self.read_run_data(eps,min_sample,update=True)
        self.source = ColumnDataSource(data=self.datadict)
        self.sourcedict['newsource'] = self.source
        self.histograms(update=True)
        self.updateaxlim()
        self.JScallback()
        self.loadbutton.callback = CustomJS(args=self.sourcedict,code=self.callbackstr)
        self.loadbutton.button_type='success'
        self.loadbutton.label = 'Click to load new data'

    def updatedtype(self,attr,old,new):
        self.loadbutton.button_type='warning'
        self.loadbutton.label = 'Loading'
        dtype = nametypes[new]
        self.read_base_data(datatype=dtype)
        self.generate_average_stats(minmem=int(self.minsize.value),update=True)
        if self.allbad:
            print("Didn't find any clusters for any parameter choices with {0} this run".format(typenames[self.dtype]))
            self.loadbutton.button_type='danger'
            self.loadbutton.label = 'No new data to load'
        elif not self.allbad:
            self.selectparam.options = self.paramlist
            self.selectparam.value = self.paramchoices[self.goodind]
            eps,min_sample = [i.split('=')[-1] for i in self.selectparam.value.split(', ')]
            eps = float(eps)
            min_sample = int(min_sample)
            # read in new self.source
            self.read_run_data(eps,min_sample,update=True)
            self.source = ColumnDataSource(data=self.datadict)
            self.sourcedict['newsource'] = self.source
            self.histograms(update=True)
            self.updateaxlim()
            self.JScallback()
            self.loadbutton.callback = CustomJS(args=self.sourcedict,code=self.callbackstr)
            self.loadbutton.button_type='success'
            self.loadbutton.label = 'Click to load new data'

    def updatecase(self,attr,old,new):
        self.loadbutton.button_type='warning'
        self.loadbutton.label = 'Loading'
        self.case = new
        caselist = glob.glob('*.hdf5')
        cases = np.unique(np.array([i.split('_')[0].split('case')[-1] for i in caselist])).astype('int')
        cases.sort()
        cases = cases.astype('str')

        timelist = glob.glob('case{0}*.hdf5'.format(new))
        times = np.array([i.split('_')[1].split('.hdf5')[0] for i in timelist])[::-1]

        self.selectcase.options = list(cases)
        self.selecttime.options = list(times)
        self.selecttime.value = times[0]

    def updatetime(self,attr,old,new):
        self.loadbutton.button_type='warning'
        self.loadbutton.label = 'Loading'
        self.timestamp = new
        dtype = nametypes[self.selectdtype.value]
        self.read_base_data(case=self.case,timestamp=self.timestamp,datatype=dtype)
        self.generate_average_stats(minmem=int(self.minsize.value),update=True)
        self.selectdtype.options = self.alldtypes
        if self.allbad:
            print("Didn't find any clusters for any parameter choices with {0} this run".format(typenames[self.dtype]))
            self.loadbutton.button_type='danger'
            self.loadbutton.label = 'No new data to load'
        elif not self.allbad:
            self.selectparam.options = self.paramlist
            self.selectparam.value = self.paramchoices[self.goodind]
            eps,min_sample = [i.split('=')[-1] for i in self.selectparam.value.split(', ')]
            eps = float(eps)
            min_sample = int(min_sample)
            # read in new self.source
            self.read_run_data(eps,min_sample,update=True)
            self.source = ColumnDataSource(data=self.datadict)
            self.sourcedict['newsource'] = self.source
            self.histograms(update=True)
            self.updateaxlim()
            self.JScallback()
            self.loadbutton.callback = CustomJS(args=self.sourcedict,code=self.callbackstr)
            self.loadbutton.button_type='success'
            self.loadbutton.label = 'Click to load new data'

    def exportplots(self):
        self.saveplots.button_type='danger'
        self.saveplots.label = 'Saving plots'
        if 'histkeys' not in dir(self):
            histkeys = self.datadict.keys()
            self.histkeys = np.array([key.lower().replace(' ','_') for key in histkeys])
        xkey = self.histkeys[self.xradio.active]
        ykey = self.histkeys[self.yradio.active]
        self.exportscatter(xkey,ykey)

        pth = '{0}/hist_efficiency'.format(resultpath)
        if not os.path.exists(pth):
            os.system('mkdir -p {0}'.format(pth))
        fname = '{0}/case{1}_eps{2}_min{3}_{4}.svg'.format(pth,self.case,self.epsval,self.minval,self.timestamp)
        if not os.path.exists(fname):
            export_svgs(self.p_efficiency,filename=fname)
            if inkscape:
                os.system('{0} --without-gui {1} --export-pdf={2}'.format(inkscape,fname,fname.replace('.svg','.pdf')))
                os.system('rm {0}'.format(fname))

        pth = '{0}/hist_completeness'.format(resultpath)
        if not os.path.exists(pth):
            os.system('mkdir -p {0}'.format(pth))
        fname = '{0}/case{1}_eps{2}_min{3}_{4}.svg'.format(pth,self.case,self.epsval,self.minval,self.timestamp)
        if not os.path.exists(fname):
            export_svgs(self.p_completeness,filename=fname)
            if inkscape:
                os.system('{0} --without-gui {1} --export-pdf={2}'.format(inkscape,fname,fname.replace('.svg','.pdf')))
                os.system('rm {0}'.format(fname))

        pth = '{0}/hist_found_silhouette'.format(resultpath)
        if not os.path.exists(pth):
            os.system('mkdir -p {0}'.format(pth))
        fname = '{0}/case{1}_eps{2}_min{3}_{4}.svg'.format(pth,self.case,self.epsval,self.minval,self.timestamp)
        if not os.path.exists(fname):
            export_svgs(self.p_found_silhouette,filename=fname)
            if inkscape:
                os.system('{0} --without-gui {1} --export-pdf={2}'.format(inkscape,fname,fname.replace('.svg','.pdf')))
                os.system('rm {0}'.format(fname))

        pth = '{0}/hist_matched_silhouette'.format(resultpath)
        if not os.path.exists(pth):
            os.system('mkdir -p {0}'.format(pth))
        fname = '{0}/case{1}_eps{2}_min{3}_{4}.svg'.format(pth,self.case,self.epsval,self.minval,self.timestamp)
        if not os.path.exists(fname):
            export_svgs(self.p_matched_silhouette,filename=fname)
            if inkscape:
                os.system('{0} --without-gui {1} --export-pdf={2}'.format(inkscape,fname,fname.replace('.svg','.pdf')))
                os.system('rm {0}'.format(fname))

        pth = '{0}/hist_found_size'.format(resultpath)
        if not os.path.exists(pth):
            os.system('mkdir -p {0}'.format(pth))
        fname = '{0}/case{1}_eps{2}_min{3}_{4}.svg'.format(pth,self.case,self.epsval,self.minval,self.timestamp)
        if not os.path.exists(fname):
            export_svgs(self.p_found_size,filename=fname)
            if inkscape:
                os.system('{0} --without-gui {1} --export-pdf={2}'.format(inkscape,fname,fname.replace('.svg','.pdf')))
                os.system('rm {0}'.format(fname))

        pth = '{0}/hist_matched_size'.format(resultpath)
        if not os.path.exists(pth):
            os.system('mkdir -p {0}'.format(pth))
        fname = '{0}/case{1}_eps{2}_min{3}_{4}.svg'.format(pth,self.case,self.epsval,self.minval,self.timestamp)
        if not os.path.exists(fname):
            export_svgs(self.p_matched_size,filename=fname)
            if inkscape:
                os.system('{0} --without-gui {1} --export-pdf={2}'.format(inkscape,fname,fname.replace('.svg','.pdf')))
                os.system('rm {0}'.format(fname))

        self.saveplots.button_type='primary'
        self.saveplots.label = 'Save plots'

    def exportscatter(self,xkey,ykey):
        pth = '{0}/scatter_linear'.format(resultpath)
        if not os.path.exists(pth):
            os.system('mkdir -p {0}'.format(pth))
        fname = '{0}/{1}_vs_{2}_case{3}_eps{4}_min{5}_{6}.svg'.format(pth,xkey,ykey,self.epsval,self.minval,self.case,self.timestamp)
        if not os.path.exists(fname):
            export_svgs(self.p1,filename=fname)
            if inkscape:
                os.system('{0} --without-gui {1} --export-pdf={2}'.format(inkscape,fname,fname.replace('.svg','.pdf')))
                os.system('rm {0}'.format(fname))

        pth = '{0}/scatter_semilogx'.format(resultpath)
        if not os.path.exists(pth):
            os.system('mkdir -p {0}'.format(pth))
        fname = '{0}/{1}_vs_{2}_case{3}_eps{4}_min{5}_{6}.svg'.format(pth,xkey,ykey,self.epsval,self.minval,self.case,self.timestamp)
        if not os.path.exists(fname):
            export_svgs(self.p2,filename=fname)
            if inkscape:
                os.system('{0} --without-gui {1} --export-pdf={2}'.format(inkscape,fname,fname.replace('.svg','.pdf')))
                os.system('rm {0}'.format(fname))

        pth = '{0}/scatter_semilogy'.format(resultpath)
        if not os.path.exists(pth):
            os.system('mkdir -p {0}'.format(pth))
        fname = '{0}/{1}_vs_{2}_case{3}_eps{4}_min{5}_{6}.svg'.format(pth,xkey,ykey,self.epsval,self.minval,self.case,self.timestamp)
        if not os.path.exists(fname):
            export_svgs(self.p3,filename=fname)
            if inkscape:
                os.system('{0} --without-gui {1} --export-pdf={2}'.format(inkscape,fname,fname.replace('.svg','.pdf')))
                os.system('rm {0}'.format(fname))

        pth = '{0}/scatter_loglog'.format(resultpath)
        if not os.path.exists(pth):
            os.system('mkdir -p {0}'.format(pth)) 
        fname = '{0}/{1}_vs_{2}_case{3}_eps{4}_min{5}_{6}.svg'.format(pth,xkey,ykey,self.epsval,self.minval,self.case,self.timestamp)
        if not os.path.exists(fname):
            export_svgs(self.p4,filename=fname)
            if inkscape:
                os.system('{0} --without-gui {1} --export-pdf={2}'.format(inkscape,fname,fname.replace('.svg','.pdf')))
                os.system('rm {0}'.format(fname))

    def updatestatplot(self, attr, old, new):
        num = int(new)
        self.generate_average_stats(minmem=num)
        for dtype in self.alldtypes:
            dtype = nametypes[dtype]
            ss = getattr(self,'{0}_statsource'.format(dtype))
            c1 = getattr(self,'{0}_c1'.format(dtype))
            c1l = getattr(self,'{0}_c1l'.format(dtype))
            c2 = getattr(self,'{0}_c2'.format(dtype))
            c3 = getattr(self,'{0}_c3'.format(dtype))
            c4 = getattr(self,'{0}_c4'.format(dtype))
            c1.data_source.data['numc'] = ss.data['numc']
            c1.glyph.y = 'numc'
            c1l.data_source.data['tnumc'] = ss.data['tnumc']
            c1l.glyph.y = 'tnumc'
            c2.data_source.data['avgeff'] = ss.data['avgeff']
            c2.glyph.y = 'avgeff'
            c3.data_source.data['avgcom'] = ss.data['avgcom']
            c3.glyph.y = 'avgcom'
            c4.data_source.data['avgfsi'] = ss.data['avgfsi']
            c4.glyph.y = 'avgfsi'


starter = display_result(case='8',timestamp='2018-07-18.12.04.04.618630',pad=0.1)

