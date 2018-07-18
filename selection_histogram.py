''' Present a scatter plot with linked histograms on both axes.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve selection_histogram.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/selection_histogram
in your browser.
'''

import h5py
import glob
import copy
import numpy as np
from clustering_stats import *

from bokeh.layouts import row, column, widgetbox
from bokeh.models import BoxSelectTool, LassoSelectTool, Spacer,CustomJS
from bokeh.models.glyphs import Circle
from bokeh.models.widgets import Toggle,RadioButtonGroup,AutocompleteInput,Tabs, Panel, Select, Button
from bokeh.plotting import figure, curdoc, ColumnDataSource, reset_output
from bokeh.io import show
from bokeh import events

case = 7

neighbours = 20

TOOLS="pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,save"

resultpath = '/Users/nat/chemtag/clustercases/'

# Make additions here if windows ever used

typenames = {'spec':'spectra','abun':'abundances'}
nametypes = {'spectra':'spec','abundances':'abun'}

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
        if datatype:
            self.dtype = datatype
        if case:
            self.case = case
        if timestamp:
            self.timestamp = timestamp
        #self.chemspace = self.data['{0}'.format(self.dtype)][:]
        #self.labels_true = self.data['labels_true'][:]
        self.tsize = self.data['true_size'][:]
        self.tsil = self.data['{0}_true_sil_neigh{1}'.format(self.dtype,neighbours)][:]
        # scrub nans
        self.tsil[np.isnan(self.tsil)]=-1
        self.labels_pred = self.data['{0}_labels_pred'.format(self.dtype)][:]
        self.numcs = []
        self.goodind = 0
        for row in range(self.labels_pred.shape[0]):
            labs = np.unique(self.labels_pred[row])
            bad = np.where(labs==-1)
            if len(bad[0])>0:
                labs = np.delete(labs,bad[0][0])
            self.numcs.append(len(labs))
        self.numcs = np.array(self.numcs)
        self.goodinds = np.where(self.numcs > 0)
        if len(self.goodinds[0]) > 0:
            self.goodind = self.goodinds[0][0]
        elif len(goodinds[0])==0:
            self.allbad = True
        self.min_samples = self.data.attrs['{0}_min'.format(self.dtype)][:]
        self.eps = self.data.attrs['{0}_eps'.format(self.dtype)][:]
        self.epsval = self.eps[self.goodind]
        self.minval = self.min_samples[self.goodind]
        self.paramchoices = []
        for i in range(len(self.eps)):
            self.paramchoices.append('eps={0}, min={1}'.format(self.eps[i],self.min_samples[i]))
        self.paramlist = list(np.array(self.paramchoices)[self.goodinds])

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

    def JScallback(self,):
        """
        Makes custom JavaScript callback from bokeh so you can easily swap source dictionaries.
        """
        self.callbackstr ="""
var sdata = source.data;
var ndata = newsource.data;
var eff = heff.data;
var com = hcom.data;
var fsi = hfsi.data;
var msi = hmsi.data;
var fsz = hfsz.data;
var msz = hmsz.data;
var neweff = newheff.data;
var newcom = newhcom.data;
var newfsi = newhfsi.data;
var newmsi = newhmsi.data;
var newfsz = newhfsz.data;
var newmsz = newhmsz.data;

for (key in ndata) {
    sdata[key] = [];
    for (i=0;i<ndata[key].length;i++){
    sdata[key].push(ndata[key][i]);
    }
}

for (key in neweff) {
    eff[key] = [];
    for (i=0;i<neweff[key].length;i++){
    eff[key].push(neweff[key][i]);
    }
}

for (key in newcom) {
    com[key] = [];
    for (i=0;i<newcom[key].length;i++){
    com[key].push(newcom[key][i]);
    }
}

for (key in newfsi) {
    fsi[key] = [];
    for (i=0;i<newfsi[key].length;i++){
    fsi[key].push(newfsi[key][i]);
    }
}

for (key in newmsi) {
    msi[key] = [];
    for (i=0;i<newmsi[key].length;i++){
    msi[key].push(newmsi[key][i]);
    }
}

for (key in newfsz) {
    fsz[key] = [];
    for (i=0;i<newfsz[key].length;i++){
    fsz[key].push(newfsz[key][i]);
    }
}

for (key in newmsz) {
    msz[key] = [];
    for (i=0;i<newmsz[key].length;i++){
    msz[key].push(newmsz[key][i]);
    }
}

source.change.emit();
heff.change.emit();
hcom.change.emit();
hfsi.change.emit();
hmsi.change.emit();
hfsz.change.emit();
hmsz.change.emit();

"""

    def layout_plots(self):
        self.read_base_data()
        if self.allbad:
            print("Didn't find any clusters for any parameter choices with {0} this run".format(typenames[self.dtype]))
        elif not self.allbad:
            self.read_run_data()
            self.center_plot()
            self.histograms()
            self.buttons()
            # Here's where you decide the distribution of plots
            self.layout = row(column(widgetbox(self.toggleline),widgetbox(self.xradio,name='x-axis'),
                            widgetbox(self.yradio,name='y-axis'),widgetbox(self.selectdtype),
                            widgetbox(self.selectparam),
                            widgetbox(self.loadbutton)),
                     column(Tabs(tabs=self.panels,width=self.sqside),row(self.p_found_size,self.p_matched_size)),
                     column(self.p_efficiency,self.p_found_silhouette),
                     column(self.p_completeness,self.p_matched_silhouette),)

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
        return [self.bcolor,self.unscolor,self.outcolor,
                self.histcolor,self.maincolor]

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
                         x_range=(xmin,xmax),y_range=(ymin,ymax))
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
                         x_range=(slxmin,xmax),y_range=(ymin,ymax))
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
                         x_range=(xmin,xmax), y_range=(slymin,ymax))
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
                         x_range=(slxmin,xmax), y_range=(slymin,ymax))
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
        print(bins)
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
                        y_axis_type=yscale)
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
        self.make_hist('Found Silhouette',bins=np.linspace(-1,1,2*nbins),
                  x_range=(0,1),yscale='log',update=update)
        self.make_hist('Matched Silhouette',bins=np.linspace(-1,1,2*nbins),
                  x_range=(0,1),yscale='log',background=self.tsil,
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

        self.labels = list(self.source.data.keys())

        self.xradio = RadioButtonGroup(labels=self.labels, active=0,name='x-axis')
        self.yradio = RadioButtonGroup(labels=self.labels, active=1,name='y-axis')

        self.selectdtype = Select(title='data type',value='spectra',options=list(nametypes.keys()))
        self.selectdtype.on_change('value',self.updatedtype)

        
        
        self.selectparam = Select(title="parameter values", value=self.paramchoices[self.goodind], 
                           options=self.paramlist)
        self.selectparam.on_change('value',self.updateparam)
        self.loadbutton = Button(label='Load New Data', button_type='success')
        self.JScallback()
        self.loadbutton.callback = CustomJS(args=self.sourcedict,code=self.callbackstr)
        
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
        axlims,lineparams = findextremes(self.source.data[self.labels[self.xradio.active]],
                                         self.source.data[self.labels[self.yradio.active]],
                                         pad=self.pad)
        xmin,xmax,ymin,ymax=axlims
        minlim,maxlim = lineparams

        if minlim < 0:
            lminlim = lzp
        else:
            lminlim = minlim

        if xmin < 0:
            slxmin = zp[self.labels[self.xradio.active]]
        else:
            slxmin = xmin

        if ymin < 0:
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

        print(self.maxsize)
        self.p_found_size.x_range.end = self.maxsize
        self.p_matched_size.x_range.end = self.maxsize

        self.l1.data_source.data['x'] = lineparams
        self.l1.data_source.data['y'] = lineparams

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

    def updatedtype(self,attr,old,new):
        self.loadbutton.button_type='warning'
        dtype = nametypes[new]
        self.read_base_data(datatype=dtype)
        self.read_run_data(update=False)
        self.selectparam.options = self.paramlist
        self.selectparam.value = self.paramchoices[self.goodind]
        self.source = ColumnDataSource(data=self.datadict)
        self.histograms(update=True)
        self.updateaxlim()
        self.sourcedict['heff'] = self.hsource_efficiency
        self.sourcedict['hcom'] = self.hsource_completeness
        self.sourcedict['hfsi'] = self.hsource_found_silhouette
        self.sourcedict['hmsi'] = self.hsource_matched_silhouette
        self.sourcedict['hfsz'] = self.hsource_found_size
        self.sourcedict['hmsz'] = self.hsource_matched_size
        self.JScallback()
        self.loadbutton.callback = CustomJS(args=self.sourcedict,code=self.callbackstr)
        self.loadbutton.button_type='success'

starter = display_result(case=8,timestamp='2018-07-18.12.04.04.618630',pad=0.1)



goodcasefiles = ['case8_2018-07-12.17.56.09.902178.hdf5',
                 'case6_2018-07-12.17.58.51.280643.hdf5',
                 'case4_2018-07-12.18.01.18.731772.hdf5',
                 'case7_2018-07-09.19.50.41.862297.hdf5']

