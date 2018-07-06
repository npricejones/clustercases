''' Present a scatter plot with linked histograms on both axes.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve selection_histogram.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/selection_histogram
in your browser.
'''

import h5py
import glob
import numpy as np
from clustering_stats import *

from bokeh.layouts import row, column, widgetbox
from bokeh.models import BoxSelectTool, LassoSelectTool, Spacer,CustomJS
from bokeh.models.glyphs import Circle
from bokeh.models.widgets import Toggle,RadioButtonGroup,AutocompleteInput,Tabs, Panel, Select
from bokeh.plotting import figure, curdoc, ColumnDataSource
from bokeh.io import show

case = 7

neighbours = 20









data={'Efficiency':efficiency,'Completeness':completeness,
      'Found Silhouette':ssil_found,'True Silhouette':ssil_true[matchtlabs],
      'Found Size':pcount,'Matched Size':tcount[matchtlabs]}

TOOLS="pan,wheel_zoom,box_select,lasso_select,reset"

resultpath = '/Users/nat/chemtag/clustercases/'

files = glob.glob('*.hdf5')

typenames = {'spec':'spectra','abun':'abundances'}

#case = AutocompleteInput(completions=)
#timestamp = AutocompleteInput(completions=)

zp = 1e-12

# create the scatter plot
panels = []

def findextremes(x,y,pad=0.1):
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    xmin -= pad*xmax
    xmax += pad*xmax
    ymin -= pad*ymax
    ymax += pad*ymax
    minlim = np.min([xmin,ymin])
    maxlim = np.max([xmax,ymax])
    return (xmin,xmax,ymin,ymax),[minlim,maxlim]

class prophist(object):
    def __init__(self,arr,bins=20):
        self.arr = arr
        self.hist, self.edges = np.histogram(arr, bins=bins)
        self.hist = self.hist.astype('float')
        self.zeros = np.zeros(len(self.edges)-1)
        self.hist_max = max(self.hist)*1.1

    def plot_hist(self,x_range=(),xlabel='',
                  xscale='linear',yscale='linear',background=[],
                  colors=['#FFF7EA','#4C230A','#280004','#F6BD60','#A53F2B']):
        backcolor,unselectcolor,outlinecolor,mainhistcolor,mainptcolor =  colors
        if background != []:
            self.backhist = np.histogram(background,bins=self.edges)[0]
            self.backhist = self.backhist.astype('float')
            backmax = np.max(self.backhist)*1.1
            if backmax > self.hist_max:
                self.hist_max = backmax
        ymax = self.hist_max
        if x_range==():
            x_range = (np.min(self.edges),np.max(self.edges))
        if yscale=='linear':
            ymin = 0
        elif yscale=='log':
            ymin=plot_eps+plot_eps/2.
            self.hist[self.hist < plot_eps] = plot_eps
            if background != []:
                self.backhist[self.backhist < plot_eps] = plot_eps
        self.pt = figure(toolbar_location=None, plot_width=220, plot_height=200, x_range=x_range,
                    y_range=(ymin, self.hist_max), min_border=10, min_border_left=50, 
                    y_axis_location="right",x_axis_label=xlabel,
                    x_axis_type=xscale,y_axis_type=yscale)
        self.pt.xgrid.grid_line_color = None
        #pt.yaxis.major_label_orientation = np.pi/4
        self.pt.background_fill_color = backcolor

        if background != []:
            self.pt.quad(bottom=ymin, left=self.edges[:-1], right=self.edges[1:], 
                         top=self.backhist, color=unselectcolor, line_color=outlinecolor)
        self.pt.quad(bottom=ymin, left=self.edges[:-1], right=self.edges[1:], 
                     top=self.hist, color=mainhistcolor, line_color=outlinecolor,alpha=0.7)
        self.h1 = self.pt.quad(bottom=ymin, left=self.edges[:-1], right=self.edges[1:], 
                               top=self.zeros, alpha=0.6, color=mainptcolor,line_color=None)

class read_results(object):

    def __init__(self,ind = 0, datatype = 'spec', case = 7, 
                 timestamp = '2018-06-29.20.20.13.729481'):
        self.ind = ind
        self.dtype = datatype
        self.case = case
        self.timestamp = timestamp

    def read_base_data(self,datatype=None,neighbours=20):
        self.data = h5py.File('case{0}_{1}.hdf5'.format(self.case,self.timestamp),'r+')
        if datatype:
            self.dtype = datatype
        #self.chemspace = self.data['{0}'.format(self.dtype)][:]
        #self.labels_true = self.data['labels_true'][:]
        self.tsize = self.data['true_size'][:]
        self.tsil = self.data['{0}_true_sil_neigh{1}'.format(self.dtype,neighbours)][:]
        #self.labels_pred = self.data['{0}_labels_pred'.format(self.dtype)][:]
        self.min_samples = self.data.attrs['{0}_min'.format(self.dtype)][:]
        self.eps = self.data.attrs['{0}_eps'.format(self.dtype)][:]

    def find_ind(self,eps,min_samples):
        return np.where((eps==self.eps) & (min_samples==self.min_samples))

    def read_run_data(self,ind=None, eps=None, min_samples=None, neighbours=20):
        if ind:
            self.ind = ind
        if eps and min_samples:
            self.ind = self.find_ind()[0][0]
        eps = self.eps[self.ind]
        min_samples = self.min_samples[self.ind]
        self.msil = self.tsil[matchtlabs]
        self.fsil = self.data['{0}_found_sil_eps{1}_min{2}_neigh{3}'.format(self.dtype,eps,min_samples,neighbours)][:]
        self.eff = self.data['{0}_eff_eps{1}_min{2}'.format(self.dtype,eps,min_samples)][:]
        self.com = self.data['{0}_com_eps{1}_min{2}'.format(self.dtype,eps,min_samples)][:]
        self.fsize = self.data['{0}_found_size_eps{1}_min{2}'.format(self.dtype,eps,min_samples)][:]
        self.msize = self.tsize[matchtlabs]
        self.numc = len(self.fsize)
        self.datadict = {'Efficiency':self.eff,'Completeness':self.com,
                         'Found Silhouette':self.fsil,'True Silhouette':self.tsil
                         'Found Size':self.fsize,'Matched Size':self.msize}

class display_result(read_results):

    def __init__(self,ind = 0, datatype = 'spec', case = 7, 
                 timestamp = '2018-06-29.20.20.13.729481', pad = 0.1, 
                 backgroundhist = True,sqside = 460,tools=TOOLS):
        read_results.__init__(self,ind=ind,datatype=datatype,case=case,
                              timestamp=timestamp)
        self.sqsize=sqside
        self.backgroundhist=backgroundhist
        self.tools=tools
        self.set_colors()
        self.read_base_data()
        self.read_run_data()
        self.center_plot()
        self.histograms()
        self.buttons()
        #Spacer(width=50, height=100)
        layout = row(column(widgetbox(self.toggleline),widgetbox(self.xradio,name='x-axis'),
                            widgetbox(self.yradio,name='y-axis'),widgetbox(self.selecteps),widgetbox(self.selectmin)),
                     column(Tabs(tabs=self.panels,width=self.sqside),row(self.pt3.pt,self.pb3.pt)),
                     column(self.pt1.pt,self.pb1.pt),
                     column(self.pt2.pt,self.pb2.pt),)

        curdoc().add_root(layout)
        curdoc().title = "DBSCAN on {0} with eps={1}, min_samples={2}".format(typenames[self.dtype],
                                                                              self.eps[self.ind],
                                                                              self.min_samples[self.ind])

    def set_colors(self):
        self.bcolor = "#FFF7EA" #cream
        self.unscolor = "#4C230A" #dark brown
        self.maincolor = "#A53F2B" #dark red
        self.histcolor = "#F6BD60" #yellow
        self.outcolor = "#280004" #dark red black
        return [self.bcolor,self.unscolor,self.outcolor,self.histcolor,self.maincolor]

    def center_plot(self,x=None,y=None):
        self.panels = []
        if not x:
            x = self.efficiency
            xlabel = 'Efficiency'
        elif x:
            xlabel = x
            x = self.datadict[xlabel]
        if not y:
            y = self.completeness
            ylabel = 'Completeness'
        elif y:
            ylabel = y
            y = self.datadict[ylabel]
        axlims,lineparams = findextremes(x,y)
        xmin,xmax,ymin,ymax=axlims
        minlim,maxlim = lineparams

        if minlim < 0:
            lminlim = zp
        else:
            lminlim = minlim

        # LINEAR TAB
        self.p1 = figure(tools=TOOLS, plot_width=sqside, plot_height=sqside, 
                         min_border=10, min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left',
                         x_axis_label=xlabel,y_axis_label=ylabel,
                         x_axis_type='linear',y_axis_type='linear',
                         x_range=(xmin,xmax),y_range=(xmin,xmax))
        self.p1.background_fill_color = self.bcolor
        self.p1.select(BoxSelectTool).select_every_mousemove = False
        self.p1.select(LassoSelectTool).select_every_mousemove = False

        self.r1 = self.p1.scatter(x, y, size=3, color=mainptcolor, alpha=0.6)
        self.l1 = self.p1.line(lineparams,lineparams,
                               color=self.outcolor)
        self.r1.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        self.panels.append(Panel(child=self.p1,title='linear'))


        # SEMILOGX TAB
        if xmin < 0:
            slxmin = zp
        else:
            slxmin = xmin
        self.p2 = figure(tools=TOOLS, plot_width=sqside, plot_height=sqside, 
                         min_border=10, min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left',
                         x_axis_label=xlabel,y_axis_label=ylabel,
                         x_axis_type='log',y_axis_type='linear',
                         x_range=(slxmin,xmax),y_range=(ymin,ymax))
        self.p2.background_fill_color = self.bcolor
        self.p2.select(BoxSelectTool).select_every_mousemove = False
        self.p2.select(LassoSelectTool).select_every_mousemove = False

        self.r2 = self.p2.scatter(x, y, size=3, color=mainptcolor, alpha=0.6)
    
        self.l2 = self.p2.line([lminlim,maxlim],[lminlim,maxlim],
                               color=self.outcolor)
        self.r2.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        self.panels.append(Panel(child=self.p2,title='semilogx'))


        # SEMILOGY TAB
        if ymin < 0:
            slymin = zp
        else:
            slymin = ymin
        self.p3 = figure(tools=TOOLS, plot_width=sqside, plot_height=sqside, 
                         min_border=10, min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left',
                         x_axis_label=xlabel,y_axis_label=ylabel,
                         x_axis_type='linear',y_axis_type='log',
                         x_range=(xmin,xmax),y_range=(slymin,ymax))
        self.p3.background_fill_color = self.bcolor
        self.p3.select(BoxSelectTool).select_every_mousemove = False
        self.p3.select(LassoSelectTool).select_every_mousemove = False

        self.r3 = self.p3.scatter(x, y, size=3, color=mainptcolor, alpha=0.6)
  
        self.l3 = self.p3.line([lminlim,maxlim],[lminlim,maxlim],
                               color=self.outcolor)
        self.r3.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        self.panels.append(Panel(child=self.p3,title='semilogy'))


        # LOG PLOT
        self.p4 = figure(tools=TOOLS, plot_width=sqside, plot_height=sqside, 
                         min_border=10, min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left',
                         x_axis_label=xlabel,y_axis_label=ylabel,
                         x_axis_type='log',y_axis_type='log',
                         x_range=(slxmin,xmax),y_range=(slymin,ymax))
        self.p4.background_fill_color = self.bcolor
        self.p4.select(BoxSelectTool).select_every_mousemove = False
        self.p4.select(LassoSelectTool).select_every_mousemove = False

        self.r4 = self.p4.scatter(x, y, size=3, color=mainptcolor, alpha=0.6)
        self.l4 = self.p4.line([lminlim,maxlim],[lminlim,maxlim],
                               color=self.outcolor)
        self.r4.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        self.panels.append(Panel(child=self.p4,title='log'))

        self.ps = [p1,p2,p3,p4]
        self.rs = [r1,r2,r3,r4]
        self.ls = [l1,l2,l3,l4]

        # NEEDS CHECK FOR <= 0 POINTS

    def histograms(self):
        self.pt1 = prophist(self.eff,bins=np.linspace(0,1,20))
        self.pt1.plot_hist(x_range=(0,1),xlabel='Efficiency',yscale='log')
        self.pb1 = prophist(self.com,bins=np.linspace(0,1,20))
        self.pb1.plot_hist(x_range=(0,1),xlabel='Completeness',yscale='log')
        self.pt2 = prophist(self.fsil,bins=np.linspace(-1,1,40))
        self.pt2.plot_hist(x_range=(0,1),xlabel='Found Silhouette',yscale='log')
        self.pb2 = prophist(self.msil,self.,bins=np.linspace(-1,1,40),background=self.tsil)
        self.pb2.plot_hist(x_range=(0,1),xlabel='True Silhouette',yscale='log')
        self.pt3 = prophist(self.fsize,bins = np.logspace(0,3,20))
        self.pt3.plot_hist(xlabel='Found Size',xscale='log',yscale='log',background=self.tsize)
        self.pb3 = prophist(self.msize,bins = np.logspace(0,3,20))
        self.pb3.plot_hist(xlabel='Matched Size',xscale='log',yscale='log',background=self.tsize)
        self.proplist = [self.pt1,self.pb1,self.pt2,self.pb2,self.pt3,self.pb3]


    def buttons(self):
        self.labels = list(self.datadict.keys())

        self.xradio = RadioButtonGroup(labels=self.labels, active=0,name='x-axis')
        self.yradio = RadioButtonGroup(labels=self.labels, active=1,name='y-axis')

        streps = list(np.unique(self.eps).astype(str))
        strmin = list(np.unique(self.min_samples).astype(str))
        self.selecteps = Select(title="eps value", value=str(self.eps[self.ind]), 
                           options=streps)
        self.selectmin = Select(title="min_samples value", value=str(self.min_samples[self.ind]), 
                           options=strmin)

        code = '''\
        object1.visible = toggle.active
        object2.visible = toggle.active
        object3.visible = toggle.active
        object4.visible = toggle.active
        '''
        linecb = CustomJS.from_coffeescript(code=code, args={})
        self.toggleline = Toggle(label="One-to-one line", button_type="success", active=True,callback=linecb)
        linecb.args = {'toggle': self.toggleline, 'object1': self.l1, 'object2': self.l2, 'object3': self.l3, 'object4': self.l4}

    def updatehist(self, attr, old, new):
        inds = np.array(new['1d']['indices'])
        if len(inds) == 0 or len(inds) == self.numc:
            for i,prop in enumerate(self.proplist):
                prop.h1.data_source.data['top'] = prop.zeros
        else:
            neg_inds = np.ones_like(self.eff, dtype=np.bool)
            neg_inds[inds] = False
            for i,prop in enumerate(props):
                hist = (np.histogram(prop.arr[inds],bins=prop.edges)[0]).astype('float')
                prop.h1.data_source.data['top'] = hist

    def updateallx(new):
        newx = self.datadict[self.labels[new]]
        axlims,lineparams = findextremes(newx,self.r1.data_source.data['y'])
        xmin,xmax,ymin,ymax=axlims
        minlim,maxlim = lineparams

        if minlim < 0:
            lminlim = zp
        else:
            lminlim = minlim

        if xmin < 0:
            slxmin = zp
        else:
            slxmin = xmin

        p1.x_range.start=xmin
        p1.x_range.end=xmax

        p2.x_range.start = slxmin
        p2.x_range.end = xmax

        p3.x_range.start = xmin
        p3.x_range.end = xmax

        p4.x_range.start = slxmin
        p4.x_range.end = xmax

        for r in self.rs:
            r.data_source.data['x'] = newx 

        l1.data_source.data['x'] = lineparams
        l1.data_source.data['y'] = lineparams

        l2.data_source.data['x'] = [lminlim,maxlim]
        l2.data_source.data['y'] = lineparams

        l3.data_source.data['x'] = lineparams
        l3.data_source.data['y'] = [lminlim,maxlim]

        l4.data_source.data['x'] = [lminlim,maxlim]
        l4.data_source.data['y'] = [lminlim,maxlim]
    
        for p in self.ps:
            p.xaxis.axis_label = self.labels[new]

    def updateallx(new):
        newy = self.datadict[self.labels[new]]
        axlims,lineparams = findextremes(self.r1.data_source.data['x'],newy)
        xmin,xmax,ymin,ymax=axlims
        minlim,maxlim = lineparams

        if minlim < 0:
            lminlim = zp
        else:
            lminlim = minlim

        if ymin < 0:
            slymin = zp
        else:
            slymin = xmin

        p1.y_range.start=ymin
        p1.y_range.end=ymax

        p2.y_range.start = ymin
        p2.y_range.end = ymax

        p3.y_range.start = slymin
        p3.y_range.end = ymax

        p4.y_range.start = slymin
        p4.y_range.end = ymax

        for r in self.rs:
            r.data_source.data['y'] = newy 

        l1.data_source.data['x'] = lineparams
        l1.data_source.data['y'] = lineparams

        l2.data_source.data['x'] = [lminlim,maxlim]
        l2.data_source.data['y'] = lineparams

        l3.data_source.data['x'] = lineparams
        l3.data_source.data['y'] = [lminlim,maxlim]

        l4.data_source.data['x'] = [lminlim,maxlim]
        l4.data_source.data['y'] = [lminlim,maxlim]
    
        for p in self.ps:
            p.yaxis.axis_label = self.labels[new]


plot_eps = 0.5
def updateeps(attr,old,new):
    epsval = float(selecteps.value)
    minval = float(selectmin.value)
    runind = np.where((eps==epsval) & (min_samples==minval))[0]



r1.data_source.on_change('selected', updatehist)
xradio.on_click(updateallx)
yradio.on_click(updateally)
selecteps.on_change

data.close()
#toggleline.on_click()
