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



#case = AutocompleteInput(completions=)
#timestamp = AutocompleteInput(completions=)

zp = 1e-12

# create the scatter plot
panels = []

class read_results(object):

    def __init__(self,ind = 0, datatype = 'spec', case = 7, 
                 timestamp = '2018-06-29.20.20.13.729481'):
        self.ind = ind
        self.dtype = datatype
        self.case = case
        self.timestamp = timestamp

    def read_base_data(self,datatype=None):
        self.data = h5py.File('case{0}_{1}.hdf5'.format(self.case,self.timestamp),'r+')
        if datatype:
            self.dtype = datatype
        #self.chemspace = self.data['{0}'.format(self.dtype)][:]
        #self.labels_true = self.data['labels_true'][:]
        self.tsize = self.data['true_size'][:]
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
        self.tsil = self.data['{0}_true_sil_neigh{1}'.format(self.dtype,neighbours)][:]
        self.fsil = self.data['{0}_found_sil_eps{1}_min{2}_neigh{3}'.format(self.dtype,eps,min_samples,neighbours)][:]
        self.eff = self.data['{0}_eff_eps{1}_min{2}'.format(self.dtype,eps,min_samples)][:]
        self.com = self.data['{0}_com_eps{1}_min{2}'.format(self.dtype,eps,min_samples)][:]
        self.fsize = self.data['{0}_found_size_eps{1}_min{2}'.format(self.dtype,eps,min_samples)][:]
        self.msize = self.data['{0}_match_size_eps{1}_min{2}'.format(self.dtype,eps,min_samples)][:]
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

    def set_colors(self):
        self.bcolor = "#FFF7EA" #cream
        self.unscolor = "#4C230A" #dark brown
        self.maincolor = "#A53F2B" #dark red
        self.histcolor = "#F6BD60" #yellow
        self.outcolor = "#280004" #dark red black

    def center_plot(x,y):
        panels = []
        self.p1 = figure(tools=TOOLS, plot_width=sqside, plot_height=sqside, 
                         min_border=10, min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left',
                         title="Linked Histograms",
                         x_axis_label='Efficiency',y_axis_label='Completeness',
                         x_axis_type='linear',y_axis_type='linear',
                         x_range=(-pad,1+pad),y_range=(-pad,1+pad))
        self.p1.background_fill_color = self.bcolor
        self.p1.select(BoxSelectTool).select_every_mousemove = False
        self.p1.select(LassoSelectTool).select_every_mousemove = False

        self.r1 = self.p1.scatter(efficiency, completeness, 
                                  size=3, color=mainptcolor, alpha=0.6)
        minlim = np.min([np.min(self.r1.data_source.data['x']),
                         np.min(self.r1.data_source.data['y'])])
        maxlim = np.max([np.max(self.r1.data_source.data['x']),
                         np.max(self.r1.data_source.data['y'])])
        self.l1 = self.p1.line([minlim,maxlim],[minlim,maxlim],
                               color=self.outcolor)
        self.r1.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        panels.append(Panel(child=self.p1,title='linear'))

        self.p2 = figure(tools=TOOLS, plot_width=sqside, plot_height=sqside, 
                         min_border=10, min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left',
                         title="Linked Histograms",
                         x_axis_label='Efficiency',y_axis_label='Completeness',
                         x_axis_type='log',y_axis_type='linear',
                         x_range=(zp,1+pad),y_range=(-pad,1+pad))
        self.p2.background_fill_color = self.bcolor
        self.p2.select(BoxSelectTool).select_every_mousemove = False
        self.p2.select(LassoSelectTool).select_every_mousemove = False

        self.r2 = self.p2.scatter(efficiency, completeness, 
                                  size=3, color=mainptcolor, alpha=0.6)
        minlim = np.min([np.min(self.r2.data_source.data['x']),
                         np.min(self.r2.data_source.data['y'])])
        maxlim = np.max([np.max(self.r2.data_source.data['x']),
                         np.max(self.r2.data_source.data['y'])])
        self.l2 = self.p2.line([minlim,maxlim],[minlim,maxlim],
                               color=self.outcolor)
        self.r2.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        panels.append(Panel(child=self.p2,title='semilogx'))

        self.p3 = figure(tools=TOOLS, plot_width=sqside, plot_height=sqside, 
                         min_border=10, min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left',
                         title="Linked Histograms",
                         x_axis_label='Efficiency',y_axis_label='Completeness',
                         x_axis_type='linear',y_axis_type='log',
                         x_range=(-pad,1+pad),y_range=(zp,1+pad))
        self.p3.background_fill_color = self.bcolor
        self.p3.select(BoxSelectTool).select_every_mousemove = False
        self.p3.select(LassoSelectTool).select_every_mousemove = False

        self.r3 = self.p3.scatter(efficiency, completeness, 
                                  size=3, color=mainptcolor, alpha=0.6)
        minlim = np.min([np.min(self.r3.data_source.data['x']),
                         np.min(self.r3.data_source.data['y'])])
        maxlim = np.max([np.max(self.r3.data_source.data['x']),
                         np.max(self.r3.data_source.data['y'])])
        self.l3 = self.p3.line([minlim,maxlim],[minlim,maxlim],
                               color=self.outcolor)
        self.r3.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        panels.append(Panel(child=self.p3,title='semilogy'))


        self.p4 = figure(tools=TOOLS, plot_width=sqside, plot_height=sqside, 
                         min_border=10, min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left',
                         title="Linked Histograms",
                         x_axis_label='Efficiency',y_axis_label='Completeness',
                         x_axis_type='log',y_axis_type='log',
                         x_range=(zp,1+pad),y_range=(zp,1+pad))
        self.p4.background_fill_color = self.bcolor
        self.p4.select(BoxSelectTool).select_every_mousemove = False
        self.p4.select(LassoSelectTool).select_every_mousemove = False

        self.r4 = self.p4.scatter(efficiency, completeness, 
                                  size=3, color=mainptcolor, alpha=0.6)
        minlim = np.min([np.min(self.r4.data_source.data['x']),
                         np.min(self.r4.data_source.data['y'])])
        maxlim = np.max([np.max(self.r4.data_source.data['x']),
                         np.max(self.r4.data_source.data['y'])])
        self.l4 = self.p4.line([minlim,maxlim],[minlim,maxlim],
                               color=self.outcolor)
        self.r4.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        panels.append(Panel(child=self.p4,title='log'))

        # NEEDS CHECK FOR <= 0 POINTS

        p2 = figure(tools=TOOLS, plot_width=sqside, plot_height=sqside, min_border=10, min_border_left=50,
                   toolbar_location="above", x_axis_location='below', y_axis_location='left',
                   title="Linked Histograms",x_axis_label='Efficiency',y_axis_label='Completeness',
                   x_axis_type='log',y_axis_type='linear',
                   x_range=(zp,1+pad),y_range=(-pad,1+pad))
        p2.background_fill_color = backcolor
        p2.select(BoxSelectTool).select_every_mousemove = False
        p2.select(LassoSelectTool).select_every_mousemove = False

        r2 = p2.scatter(efficiency, completeness, size=3, color=mainptcolor, alpha=0.6)
        minlim = np.min([np.min(r2.data_source.data['x']),np.min(r2.data_source.data['y'])])
        maxlim = np.max([np.max(r2.data_source.data['x']),np.max(r2.data_source.data['y'])])
        l2 = p2.line([minlim,maxlim],[minlim,maxlim],color=outlinecolor)
        r2.nonselection_glyph = Circle(fill_color=unselectcolor, fill_alpha=0.1, line_color=None)

        panels.append(Panel(child=p2,title='semilogx'))

        p3 = figure(tools=TOOLS, plot_width=sqside, plot_height=sqside, min_border=10, min_border_left=50,
                   toolbar_location="above", x_axis_location='below', y_axis_location='left',
                   title="Linked Histograms",x_axis_label='Efficiency',y_axis_label='Completeness',
                   x_axis_type='linear',y_axis_type='log',
                   x_range=(-pad,1+pad),y_range=(zp,1+pad))
        p3.background_fill_color = backcolor
        p3.select(BoxSelectTool).select_every_mousemove = False
        p3.select(LassoSelectTool).select_every_mousemove = False

        r3 = p3.scatter(efficiency, completeness, size=3, color=mainptcolor, alpha=0.6)
        minlim = np.min([np.min(r3.data_source.data['x']),np.min(r3.data_source.data['y'])])
        maxlim = np.max([np.max(r3.data_source.data['x']),np.max(r3.data_source.data['y'])])
        l3 = p3.line([minlim,maxlim],[minlim,maxlim],color=outlinecolor)
        r3.nonselection_glyph = Circle(fill_color=unselectcolor, fill_alpha=0.1, line_color=None)

        panels.append(Panel(child=p3,title='semilogy'))

        p4 = figure(tools=TOOLS, plot_width=sqside, plot_height=sqside, min_border=10, min_border_left=50,
                   toolbar_location="above", x_axis_location='below', y_axis_location='left',
                   title="Linked Histograms",x_axis_label='Efficiency',y_axis_label='Completeness',
                   x_axis_type='log',y_axis_type='log',
                   x_range=(zp,1+pad),y_range=(zp,1+pad))
        p4.background_fill_color = backcolor
        p4.select(BoxSelectTool).select_every_mousemove = False
        p4.select(LassoSelectTool).select_every_mousemove = False

        r4 = p4.scatter(efficiency, completeness, size=3, color=mainptcolor, alpha=0.6)
        minlim = np.min([np.min(r4.data_source.data['x']),np.min(r4.data_source.data['y'])])
        maxlim = np.max([np.max(r4.data_source.data['x']),np.max(r4.data_source.data['y'])])
        l4 = p4.line([minlim,maxlim],[minlim,maxlim],color=outlinecolor)
        r4.nonselection_glyph = Circle(fill_color=unselectcolor, fill_alpha=0.1, line_color=None)

        panels.append(Panel(child=p4,title='log'))

plot_eps = 0.5

LINE_ARGS = dict(color=mainptcolor, line_color=None)

class prophist(object):
    def __init__(self,arr,bins=20):
        self.arr = arr
        self.hist, self.edges = np.histogram(arr, bins=bins)
        self.hist = self.hist.astype('float')
        # self.hist = np.log10(self.hist)
        # self.hist[np.isnan(self.hist)] = -1
        # self.hist[np.isinf(self.hist)] = -1
        self.zeros = np.zeros(len(self.edges)-1)
        self.hist_max = max(self.hist)*1.1

    def plot_hist(self,x_range=(),xlabel='',line=LINE_ARGS,xscale='linear',yscale='linear',background=[]):
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
                    y_range=(ymin, self.hist_max), min_border=10, min_border_left=50, y_axis_location="right",
                    x_axis_label=xlabel,x_axis_type=xscale,y_axis_type=yscale)
        self.pt.xgrid.grid_line_color = None
        #pt.yaxis.major_label_orientation = np.pi/4
        self.pt.background_fill_color = backcolor

        if background != []:
            # self.backhist = np.log10(self.backhist)
            # self.backhist[np.isnan(self.backhist)] = -1
            # self.backhist[np.isinf(self.backhist)] = -1
            self.pt.quad(bottom=ymin, left=self.edges[:-1], right=self.edges[1:], top=self.backhist, color=unselectcolor, line_color=outlinecolor)
        self.pt.quad(bottom=ymin, left=self.edges[:-1], right=self.edges[1:], top=self.hist, color=mainhistcolor, line_color=outlinecolor,alpha=0.7)
        self.h1 = self.pt.quad(bottom=ymin, left=self.edges[:-1], right=self.edges[1:], top=self.zeros, alpha=0.6, **line)

pt1 = prophist(efficiency,bins=np.linspace(0,1,20))
pt1.plot_hist(x_range=(0,1),xlabel='Efficiency',yscale='log')
pb1 = prophist(completeness,bins=np.linspace(0,1,20))
pb1.plot_hist(x_range=(0,1),xlabel='Completeness',yscale='log')
pt2 = prophist(ssil_found,bins=np.linspace(-1,1,40))
pt2.plot_hist(x_range=(0,1),xlabel='Found Silhouette',yscale='log')
pb2 = prophist(ssil_true,bins=np.linspace(-1,1,40))
pb2.plot_hist(x_range=(0,1),xlabel='True Silhouette',yscale='log')
pt3 = prophist(pcount,bins = np.logspace(0,3,20))
pt3.plot_hist(xlabel='Found Size',xscale='log',yscale='log',background=tcount)
pb3 = prophist(tcount[matchtlabs],bins = np.logspace(0,3,20))
pb3.plot_hist(xlabel='Matched Size',xscale='log',yscale='log',background=tcount)

props = [pt1,pb1,pt2,pb2,pt3,pb3]

labels = list(data.keys())

xradio = RadioButtonGroup(labels=labels, active=0,name='x-axis')
yradio = RadioButtonGroup(labels=labels, active=1,name='y-axis')

streps = list(np.unique(eps).astype(str))
strmin = list(np.unique(min_samples).astype(str))
selecteps = Select(title="eps value", value=streps[0], options=streps)
selectmin = Select(title="min_samples value", value=strmin[0], options=strmin)

code = '''\
object1.visible = toggle.active
object2.visible = toggle.active
object3.visible = toggle.active
object4.visible = toggle.active
'''
linecb = CustomJS.from_coffeescript(code=code, args={})
toggleline = Toggle(label="One-to-one line", button_type="success", active=True,callback=linecb)
linecb.args = {'toggle': toggleline, 'object1': l1, 'object2': l2, 'object3': l3, 'object4': l4}

#Spacer(width=50, height=100)
layout = row(column(widgetbox(toggleline),widgetbox(xradio,name='x-axis'),
                    widgetbox(yradio,name='y-axis'),widgetbox(selecteps),widgetbox(selectmin)),
             column(Tabs(tabs=panels,width=sqside),row(pt3.pt,pb3.pt)),column(pt1.pt,pb1.pt),column(pt2.pt,pb2.pt),)

curdoc().add_root(layout)
curdoc().title = "Selection Histogram"

# add other tabs updating
def updatehist(attr, old, new):
    inds = np.array(new['1d']['indices'])
    if len(inds) == 0 or len(inds) == len(efficiency):
        for i,prop in enumerate(props):
            prop.h1.data_source.data['top'] = prop.zeros
    else:
        neg_inds = np.ones_like(efficiency, dtype=np.bool)
        neg_inds[inds] = False
        for i,prop in enumerate(props):
            hist = (np.histogram(prop.arr[inds],bins=prop.edges)[0]).astype('float')
            # loghist = np.log10(hist)
            # loghist[np.isnan(loghist)] = -1
            # loghist[np.isinf(loghist)] = -1
            prop.h1.data_source.data['top'] = hist

ps = [p1,p2,p3,p4]
rs = [r1,r2,r3,r4]
ls = [l1,l2,l3,l4]
def updateallx(new):
    for r in rs:
        r.data_source.data['x'] = data[labels[new]] 
        xmin = np.min(r.data_source.data['x'])
        xmax = np.max(r.data_source.data['x'])
        ymin = np.min(r.data_source.data['y'])
        ymax = np.max(r.data_source.data['y'])
    xmin -= pad*xmax
    xmax += pad*xmax
    ymin -= pad*ymax
    ymax += pad*ymax
    minlim = np.min([xmin,ymin])
    maxlim = np.max([xmax,ymax])
    for l in ls:
        l.data_source.data['x'] = [minlim,maxlim]
        l.data_source.data['y'] = [minlim,maxlim]
    for p in ps:
        p.x_range.start = xmin
        p.x_range.end = xmax
        p.y_range.start = ymin
        p.y_range.end = ymax
        p.xaxis.axis_label = labels[new]

def updateally(new):
    for r in rs:
        r.data_source.data['y'] = data[labels[new]]
        xmin = np.min(r.data_source.data['x'])
        xmax = np.max(r.data_source.data['x'])
        ymin = np.min(r.data_source.data['y'])
        ymax = np.max(r.data_source.data['y'])
    xmin -= pad*xmax
    xmax += pad*xmax
    ymin -= pad*ymax
    ymax += pad*ymax
    minlim = np.min([xmin,ymin])
    maxlim = np.max([xmax,ymax])
    for l in ls:
        l.data_source.data['x'] = [minlim,maxlim]
        l.data_source.data['y'] = [minlim,maxlim]
    for p in ps:
        p.x_range.start = xmin
        p.x_range.end = xmax
        p.y_range.start = ymin
        p.y_range.end = ymax
        p.yaxis.axis_label = labels[new]

def updateeps(attr,old,new):
    epsval = float(selecteps.value)
    minval = float(selectmin.value)
    runind = np.where((eps==epsval) & (min_samples==minval))[0]
    efficiency, completeness, plabs, matchtlabs = efficiency_completeness(spec_labels_pred[runind],
                                                                          labels_true,
                                                                          minmembers=1)


r1.data_source.on_change('selected', updatehist)
xradio.on_click(updateallx)
yradio.on_click(updateally)
selecteps.on_change

data.close()
#toggleline.on_click()
