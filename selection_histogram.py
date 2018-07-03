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
from bokeh.models import BoxSelectTool, LassoSelectTool, Spacer
from bokeh.plotting import figure, curdoc, ColumnDataSource
from bokeh.models.widgets import RadioButtonGroup,AutocompleteInput
from bokeh.io import show

case = 7
timestamp = '2018-06-29.20.20.13.729481'
neighbours = 20

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
          'Found Size':pcount,'Found Silhouette':ssil_found,
          'True Silhouette':ssil_true[matchtlabs],'Matched Size':tcount[matchtlabs]}

TOOLS="pan,wheel_zoom,box_select,lasso_select,reset"

resultpath = '/Users/nat/chemtag/clustercases/'

files = glob.glob('*.hdf5')

#case = AutocompleteInput(completions=)
#timestamp = AutocompleteInput(completions=)

# create the scatter plot
p = figure(tools=TOOLS, plot_width=600, plot_height=600, min_border=10, min_border_left=50,
           toolbar_location="above", x_axis_location=None, y_axis_location=None,
           title="Linked Histograms",x_axis_label='efficiency',y_axis_label='completeness')
p.background_fill_color = "#fafafa"
p.select(BoxSelectTool).select_every_mousemove = False
p.select(LassoSelectTool).select_every_mousemove = False

r = p.scatter(efficiency, completeness, size=3, color="#3A5785", alpha=0.6)

LINE_ARGS = dict(color="#3A5785", line_color=None)

class prophist(object):
    def __init__(self,arr,bins=20):
        self.arr = arr
        self.hist, self.edges = np.histogram(arr, bins=bins)
        self.hist = np.log10(self.hist)
        self.hist[np.isnan(self.hist)] = -1
        self.hist[np.isinf(self.hist)] = -1
        self.zeros = np.zeros(len(self.edges)-1)
        self.hist_max = max(self.hist)*1.1

    def plot_hist(self,x_range=(),xlabel='',line=LINE_ARGS,xscale='linear',yscale='linear'):
        if x_range==():
            x_range = (np.min(self.edges),np.max(self.edges))

        self.pt = figure(toolbar_location=None, plot_width=300, plot_height=250, x_range=x_range,
                    y_range=(0, self.hist_max), min_border=10, min_border_left=50, y_axis_location="right",
                    x_axis_label=xlabel,x_axis_type=xscale,y_axis_type=yscale)
        self.pt.xgrid.grid_line_color = None
        #pt.yaxis.major_label_orientation = np.pi/4
        self.pt.background_fill_color = "#fafafa"

        self.pt.quad(bottom=0, left=self.edges[:-1], right=self.edges[1:], top=self.hist, color="white", line_color="#3A5785")
        self.h1 = self.pt.quad(bottom=0, left=self.edges[:-1], right=self.edges[1:], top=self.zeros, alpha=0.5, **line)

pt1 = prophist(efficiency)
pt1.plot_hist(x_range=(0,1),xlabel='efficiency')
pb1 = prophist(completeness)
pb1.plot_hist(x_range=(0,1),xlabel='completeness')
pt2 = prophist(ssil_found)
pt2.plot_hist(x_range=(0,1),xlabel='found silhouette')
pb2 = prophist(ssil_true)
pb2.plot_hist(x_range=(0,1),xlabel='true silhouette')
pt3 = prophist(pcount,bins = np.logspace(0,3,20))
pt3.plot_hist(xlabel='found size',xscale='log')
pb3 = prophist(tcount[matchtlabs],bins = np.logspace(0,3,20))
pb3.plot_hist(xlabel='true size',xscale='log')

props = [pt1,pb1,pt2,pb2,pt3,pb3]

# # create the bottom histogram
# bhist, bedges = np.histogram(completeness, bins=20)
# bzeros = np.zeros(len(bedges)-1)
# bmax = max(bhist)*1.1

# pb = figure(toolbar_location=None, plot_width=300, plot_height=250, x_range=(0, 1),
#             y_range=(0,bmax), min_border=10, y_axis_location="right",x_axis_label='completeness')
# pb.ygrid.grid_line_color = None
# #pb.xaxis.major_label_orientation = np.pi/4
# pb.background_fill_color = "#fafafa"

# pb.quad(bottom=0, left=bedges[:-1], right=bedges[1:], top=bhist, color="white", line_color="#3A5785")
# bh1 = pb.quad(bottom=0, left=bedges[:-1], right=bedges[1:], top=bzeros, alpha=0.5, **LINE_ARGS)
# bh2 = pb.quad(bottom=0, left=bedges[:-1], right=bedges[1:], top=bzeros, alpha=0.1, **LINE_ARGS)

labels = list(data.keys())

xradio = RadioButtonGroup(labels=labels, active=0)
yradio = RadioButtonGroup(labels=labels, active=1)
#Spacer(width=50, height=100)
layout = row(column(widgetbox(xradio),widgetbox(yradio)),column(p),column(pt1.pt,pb1.pt),column(pt2.pt,pb2.pt),column(pt3.pt,pb3.pt))

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
            hist = np.histogram(prop.arr[inds],bins=prop.edges)[0]
            loghist = np.log10(hist)
            loghist[np.isnan(loghist)] = -1
            loghist[np.isinf(loghist)] = -1
            prop.h1.data_source.data['top'] = loghist

def updatex(new):
    r.data_source.data['x'] = data[labels[new]] 

def updatey(new):
    r.data_source.data['y'] = data[labels[new]]

r.data_source.on_change('selected', updatehist)
xradio.on_click(updatex)
yradio.on_click(updatey)
