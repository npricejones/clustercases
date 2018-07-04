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
from bokeh.models.scales import LinearScale, LogScale
from bokeh.models.glyphs import Circle
from bokeh.models.widgets import Toggle,RadioButtonGroup,AutocompleteInput,Tabs, Panel
from bokeh.plotting import figure, curdoc, ColumnDataSource
from bokeh.io import show

case = 7
timestamp = '2018-06-29.20.20.13.729481'
neighbours = 20
pad = 0.1
backgroundhist = True

data = h5py.File('case{0}_{1}.hdf5'.format(case,timestamp),'r+')

spec = data['spec'][:]
abun = data['abun'][:]
labels_true = data['labels_true'][:]
spec_labels_pred = data['spec_labels_pred'][:]
abun_labels_pred = data['abun_labels_pred'][:]
spec_cbn = data['spec_cbn'][:]
abun_cbn = data['abun_cbn'][:]
ssil_found = data['spec_sil'][:]
ssil_true = data['true_sil'][:]

# #da = distance_metrics(abun)
# ds = distance_metrics(spec)

# # #asil = da.silhouette(abun_labels_pred[0],k=neighbours)[0]
# ssil_true = ds.silhouette(labels_true,k=neighbours)[0]

# data['true_sil'] = ssil_true

data.close()

pcount,plabs = membercount(spec_labels_pred[0])
pcount = pcount[1:]

tcount,tlabs = membercount(labels_true)

efficiency, completeness, plabs, matchtlabs = efficiency_completeness(spec_labels_pred[0],
                                                                 labels_true,
                                                                 minmembers=1)

data={'Efficiency':efficiency,'Completeness':completeness,
      'Found Silhouette':ssil_found,'True Silhouette':ssil_true[matchtlabs],
      'Found Size':pcount,'Matched Size':tcount[matchtlabs]}

scales = {'Efficiency':LinearScale,'Completeness':LinearScale,
          'Found Size':LogScale,'Found Silhouette':LinearScale,
          'True Silhouette':LinearScale,'Matched Size':LogScale}

TOOLS="pan,wheel_zoom,box_select,lasso_select,reset"

resultpath = '/Users/nat/chemtag/clustercases/'

files = glob.glob('*.hdf5')

backcolor = "#FFF7EA" #cream
unselectcolor = "#4C230A" #dark brown
mainptcolor = "#A53F2B" #dark red
mainhistcolor = "#F6BD60" #yellow
outlinecolor = "#280004" #dark red black

#case = AutocompleteInput(completions=)
#timestamp = AutocompleteInput(completions=)

zp = 1e-12
sqside = 460

# create the scatter plot
panels = []

p1 = figure(tools=TOOLS, plot_width=sqside, plot_height=sqside, min_border=10, min_border_left=50,
           toolbar_location="above", x_axis_location='below', y_axis_location='left',
           title="Linked Histograms",x_axis_label='Efficiency',y_axis_label='Completeness',
           x_axis_type='linear',y_axis_type='linear',
           x_range=(-pad,1+pad),y_range=(-pad,1+pad))
p1.background_fill_color = backcolor
p1.select(BoxSelectTool).select_every_mousemove = False
p1.select(LassoSelectTool).select_every_mousemove = False

r1 = p1.scatter(efficiency, completeness, size=3, color=mainptcolor, alpha=0.6)
minlim = np.min([np.min(r1.data_source.data['x']),np.min(r1.data_source.data['y'])])
maxlim = np.max([np.max(r1.data_source.data['x']),np.max(r1.data_source.data['y'])])
l1 = p1.line([minlim,maxlim],[minlim,maxlim],color=outlinecolor)
r1.nonselection_glyph = Circle(fill_color=unselectcolor, fill_alpha=0.1, line_color=None)

panels.append(Panel(child=p1,title='linear'))

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

eps = 0.5

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
            ymin=eps+eps/2.
            self.hist[self.hist < eps] = eps
            if background != []:
                self.backhist[self.backhist < eps] = eps
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

code = '''\
object.visible = toggle.active
'''
linecb = CustomJS.from_coffeescript(code=code, args={})
toggleline = Toggle(label="One-to-one line", button_type="success", active=True,callback=linecb)
linecb.args = {'toggle': toggleline, 'object': l1}

#Spacer(width=50, height=100)
layout = row(column(widgetbox(toggleline),widgetbox(xradio,name='x-axis'),
                    widgetbox(yradio,name='y-axis')),
             column(Tabs(tabs=panels,width=sqside),row(pt3.pt,pb3.pt)),column(pt1.pt,pb1.pt),column(pt2.pt,pb2.pt),)

curdoc().add_root(layout)
curdoc().title = "Selection Histogram"

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

def updatex(new):
    r1.data_source.data['x'] = data[labels[new]] 
    xmin = np.min(r1.data_source.data['x'])
    xmax = np.max(r1.data_source.data['x'])
    ymin = np.min(r1.data_source.data['y'])
    ymax = np.max(r1.data_source.data['y'])
    xmin -= pad*xmax
    xmax += pad*xmax
    ymin -= pad*ymax
    ymax += pad*ymax
    minlim = np.min([xmin,ymin])
    maxlim = np.max([xmax,ymax])
    l1.data_source.data['x'] = [minlim,maxlim]
    l1.data_source.data['y'] = [minlim,maxlim]
    p1.x_range.start = xmin
    p1.x_range.end = xmax
    p1.y_range.start = ymin
    p1.y_range.end = ymax
    p1.xaxis.axis_label = labels[new]
    p1.x_scale = scales[labels[new]]()

def updatey(new):
    r1.data_source.data['y'] = data[labels[new]]
    xmin = np.min(r1.data_source.data['x'])
    xmax = np.max(r1.data_source.data['x'])
    ymin = np.min(r1.data_source.data['y'])
    ymax = np.max(r1.data_source.data['y'])
    xmin -= pad*xmax
    xmax += pad*xmax
    ymin -= pad*ymax
    ymax += pad*ymax
    minlim = np.min([xmin,ymin])
    maxlim = np.max([xmax,ymax])
    l1.data_source.data['x'] = [minlim,maxlim]
    l1.data_source.data['y'] = [minlim,maxlim]
    p1.x_range.start = xmin
    p1.x_range.end = xmax
    p1.y_range.start = ymin
    p1.y_range.end = ymax
    p1.yaxis.axis_label = labels[new]
    p1.y_scale = scales[labels[new]]()

r1.data_source.on_change('selected', updatehist)
xradio.on_click(updatex)
yradio.on_click(updatey)

#toggleline.on_click()
