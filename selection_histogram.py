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
from bokeh.plotting import figure, curdoc, ColumnDataSource, reset_output
from bokeh.io import show

case = 7

neighbours = 20

TOOLS="pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,save"

resultpath = '/Users/nat/chemtag/clustercases/'

typenames = {'spec':'spectra','abun':'abundances'}

zp = 1e-3
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

class prophist(object):
    def __init__(self,obj,key,bins=20):
        self.xlabel = key
        self.arr = obj.datadict[key] 
        self.hist, self.edges = np.histogram(self.arr, bins=bins)
        self.hist = self.hist.astype('float')
        self.zeros = np.zeros(len(self.edges)-1)
        self.hist_max = max(self.hist)*1.1
        self.data={'mainhist':self.hist,'left':self.edges[:-1],'right':self.edges[1:],'zeros':self.zeros}

    def plot_hist(self,x_range=(),xscale='linear',
                  yscale='linear',background=[],update=False,
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
            self.ymin = 0
        elif yscale=='log':
            self.ymin=plot_eps+plot_eps/2.
            self.hist[self.hist < plot_eps] = plot_eps
            if background != []:
                self.backhist[self.backhist < plot_eps] = plot_eps
                self.data['backhist'] = self.backhist
        self.data['bottom'] = np.repeat(self.ymin,len(self.hist))
        self.source = ColumnDataSource(data=self.data)
        if not update:
            self.pt = figure(toolbar_location=None, plot_width=220, plot_height=200, x_range=x_range,
                        y_range=(self.ymin, self.hist_max), min_border=10, min_border_left=50, 
                        y_axis_location="right",x_axis_label=self.xlabel,
                        x_axis_type=xscale,y_axis_type=yscale)
            self.pt.xgrid.grid_line_color = None
            #pt.yaxis.major_label_orientation = np.pi/4
            self.pt.background_fill_color = backcolor

            if background != []:
                self.bghist = self.pt.quad(bottom='bottom',left='left',right='right', 
                             top='backhist', source=self.source,color=unselectcolor, line_color=outlinecolor)
            self.mainhist = self.pt.quad(bottom='bottom',left='left',right='right', 
                         top='mainhist', source=self.source, color=mainhistcolor, line_color=outlinecolor,alpha=0.7)
            self.h1 = self.pt.quad(bottom='bottom',left='left',right='right', 
                                   top='zeros',source=self.source, alpha=0.6, color=mainptcolor,line_color=None)

class read_results(object):

    def __init__(self,ind = 0, datatype = 'spec', case = 7, 
                 timestamp = '2018-07-06.14.28.11.782906'):
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

    def read_run_data(self):
        self.sourcedict = {}
        for i in range(len(self.eps)):
            self.epsval = self.eps[i]
            self.minval = self.min_samples[i]
            self.matchtlabs = self.data['{0}_match_tlabs_eps{1}_min{2}'.format(self.dtype,self.epsval,self.minval)][:]
            self.msil = self.tsil[self.matchtlabs]
            self.fsil = self.data['{0}_found_sil_eps{1}_min{2}_neigh{3}'.format(self.dtype,self.epsval,self.minval,neighbours)][:]
            self.eff = self.data['{0}_eff_eps{1}_min{2}'.format(self.dtype,self.epsval,self.minval)][:]
            self.com = self.data['{0}_com_eps{1}_min{2}'.format(self.dtype,self.epsval,self.minval)][:]
            self.fsize = self.data['{0}_found_size_eps{1}_min{2}'.format(self.dtype,self.epsval,self.minval)][:]
            self.msize = self.tsize[self.matchtlabs]
            self.numc = len(self.fsize)
            datadict = {'Efficiency':self.eff,'Completeness':self.com,
                             'Found Silhouette':self.fsil,'Matched Silhouette':self.msil,
                             'Found Size':self.fsize,'Matched Size':self.msize}
            setattr(self,'datadict_eps{0}_min{1}'.format(self.epsval,self.minval),datadict)
            #setattr(self,'source_eps{0}_min{1}'.format(self.epsval,self.minval),ColumnDataSource(data=datadict))
            self.sourcedict['source{0}'.format(i)] = ColumnDataSource(data=datadict)
            if i==0:
                self.sourcedict['source'] = ColumnDataSource(data=datadict)

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
        #self.set_colors()
        #self.layout_plots()
        

    def JScallback(self,):
        """
        Makes custom JavaScript callback from bokeh so you can easily swap source dictionaries.
        """
    
        varstr ="""
var f = cb_obj.value;
var data = source.data;
        """
        for i in range(len(self.eps)):
            varstr+='\nvar data{0} = source{0}.data'.format(i)

        actstr = '\n'

        for i in range(len(self.eps)):
            if i != len(self.eps)-1:
                actstr+="""
if (f == "source{0}") {{
for (key in data{0}) {{
    sdata[key] = [];
    for (i=0;i<data{0}[key].length;i++){{
    sdata[key].push(data{0}[key][i]);
    }}
}}
}}
            """.format(i)
            elif i == len(self.eps)-1:
                actstr+="""
if (f == "source{0}") {{
for (key in data{0}) {{
    sdata[key] = [];
    for (i=0;i<data{0}[key].length;i++){{
    sdata[key].push(data{0}[key][i]);
    }}
}}
}};
            """.format(i)
        self.callbackstr = varstr+actstr



    def layout_plots(self,update=False):
        if not update:
            self.read_base_data()
            self.read_run_data()
            self.center_plot()
            self.histograms()
            self.buttons()
            # Here's where you decide the distribution of plots
            self.layout = row(column(widgetbox(self.selectcase),widgetbox(self.selecttime),
                            widgetbox(self.toggleline),widgetbox(self.xradio,name='x-axis'),
                            widgetbox(self.yradio,name='y-axis'),widgetbox(self.selecteps),widgetbox(self.selectmin)),
                     column(Tabs(tabs=self.panels,width=self.sqside),row(self.pt3.pt,self.pb3.pt)),
                     column(self.pt1.pt,self.pb1.pt),
                     column(self.pt2.pt,self.pb2.pt),)
            curdoc().add_root(self.layout)
            curdoc().title = "DBSCAN on {0} with eps={1}, min_samples={2}".format(typenames[self.dtype], 
                                                                                  self.epsval,self.minval)
        elif update:
            for r in self.rs:
                r.data_source = self.source
                #r.glyph.x = self.labels[self.xradio.active]
                #r.glyph.y = self.labels[self.yradio.active]
            self.updateallx(self.xradio.active)
            self.updateally(self.yradio.active)
            newpt1 = prophist(self,'Efficiency',bins=np.linspace(0,1,20))
            newpt1.plot_hist(x_range=(0,1),yscale='log',update=True)
            self.pt1.mainhist.data_source = newpt1.source
            self.pt1.h1.data_source=newpt1.source
            self.pt1.hist = newpt1.hist
            self.pt1.edges = newpt1.edges
            self.pt1.zeros = newpt1.zeros
            self.pt1.hist_max = newpt1.hist_max
            self.pt1.data = newpt1.data
            #self.pt1.pt.y_range = (newpt1.ymin,self.pt1.hist_max)
            newpb1 = prophist(self,'Completeness',bins=np.linspace(0,1,20))
            newpb1.plot_hist(x_range=(0,1),yscale='log',update=True)
            self.pb1.mainhist.data_source = newpb1.source
            self.pb1.h1.data_source=newpb1.source
            self.pb1.hist = newpb1.hist
            self.pb1.edges = newpb1.edges
            self.pb1.zeros = newpb1.zeros
            self.pb1.hist_max = newpb1.hist_max
            self.pb1.data = newpb1.data
            #self.pb1.pt.y_range = (newpb1.ymin,self.pb1.hist_max)
            newpt2 = prophist(self,'Found Silhouette',bins=np.linspace(-1,1,40))
            newpt2.plot_hist(x_range=(0,1),yscale='log',update=True)
            self.pt2.mainhist.data_source = newpt2.source
            self.pt2.h1.data_source=newpt2.source
            self.pt2.hist = newpt2.hist
            self.pt2.edges = newpt2.edges
            self.pt2.zeros = newpt2.zeros
            self.pt2.hist_max = newpt2.hist_max
            self.pt2.data = newpt2.data
            #self.pt2.pt.y_range = (newpt2.ymin,self.pt2.hist_max)
            newpb2 = prophist(self,'Matched Silhouette',bins=np.linspace(-1,1,40))
            newpb2.plot_hist(x_range=(0,1),yscale='log',update=True,background=self.tsil)
            self.pb2.mainhist.data_source = newpb2.source
            self.pb2.bghist.data_source = newpb2.source
            self.pb2.h1.data_source=newpb2.source
            self.pb2.hist = newpb2.hist
            self.pb2.edges = newpb2.edges
            self.pb2.zeros = newpb2.zeros
            self.pb2.hist_max = newpb2.hist_max
            self.pb2.data = newpb2.data
            #self.pb2.pt.y_range = (newpb2.ymin,self.pb2.hist_max)
            newpt3 = prophist(self,'Found Size',bins= np.logspace(0,3,20))
            newpt3.plot_hist(xscale='log',yscale='log',update=True,background=self.tsize)
            self.pt3.mainhist.data_source = newpt3.source
            self.pt3.bghist.data_source = newpt3.source
            self.pt3.h1.data_source=newpt3.source
            self.pt3.hist = newpt3.hist
            self.pt3.edges = newpt3.edges
            self.pt3.zeros = newpt3.zeros
            self.pt3.hist_max = newpt3.hist_max
            self.pt3.data = newpt3.data
            #self.pt3.pt.y_range = (newpt3.ymin,self.pt3.hist_max)
            newpb3 = prophist(self,'Matched Size',bins= np.logspace(0,3,20))
            newpb3.plot_hist(xscale='log',yscale='log',update=True,background=self.tsize)
            self.pb3.mainhist.data_source = newpb3.source
            self.pb3.bghist.data_source = newpb3.source
            self.pb3.h1.data_source=newpb3.source
            self.pb3.hist = newpb3.hist
            self.pb3.edges = newpb3.edges
            self.pb3.zeros = newpb3.zeros
            self.pb3.hist_max = newpb3.hist_max
            self.pb3.data = newpb3.data
            #self.pb3.pt.y_range = (newpb3.ymin,self.pb3.hist_max)

        self.r1.data_source.on_change('selected', self.updatehist)
        self.r2.data_source.on_change('selected', self.updatehist)
        self.r3.data_source.on_change('selected', self.updatehist)
        self.r4.data_source.on_change('selected', self.updatehist)
        self.xradio.on_click(self.updateallx)
        self.yradio.on_click(self.updateally)
        self.selecteps.on_change('value', self.updatedata_eps)

    def set_colors(self):
        self.bcolor = "#FFF7EA" #cream
        self.unscolor = "#4C230A" #dark brown
        self.maincolor = "#A53F2B" #dark red
        self.histcolor = "#F6BD60" #yellow
        self.outcolor = "#280004" #dark red black
        return [self.bcolor,self.unscolor,self.outcolor,self.histcolor,self.maincolor]

    def center_plot(self,xlabel=None,ylabel=None):
        self.panels = []
        if not xlabel:
            xlabel = 'Efficiency'
        if not ylabel:
            ylabel = 'Completeness'
        x = self.datadict[xlabel]
        y = self.datadict[ylabel]
        axlims,lineparams = findextremes(x,y,pad=self.pad)
        xmin,xmax,ymin,ymax=axlims
        minlim,maxlim = lineparams

        if minlim < 0:
            lminlim = zp
        else:
            lminlim = minlim

        # LINEAR TAB
        self.p1 = figure(tools=TOOLS, plot_width=self.sqside, plot_height=self.sqside, 
                         min_border=10, min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left',
                         x_axis_label=xlabel,y_axis_label=ylabel,
                         x_axis_type='linear',y_axis_type='linear',
                         x_range=(xmin,xmax),y_range=(ymin,ymax))
        self.p1.background_fill_color = self.bcolor
        self.p1.select(BoxSelectTool).select_every_mousemove = False
        self.p1.select(LassoSelectTool).select_every_mousemove = False

        self.r1 = self.p1.scatter(x=xlabel, y=ylabel, source=self.source, size=3, color=self.maincolor, alpha=0.6)
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
        self.p2 = figure(tools=TOOLS, plot_width=self.sqside, plot_height=self.sqside, 
                         min_border=10, min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left',
                         x_axis_label=xlabel,y_axis_label=ylabel,
                         x_axis_type='log',y_axis_type='linear',
                         x_range=(slxmin,xmax),y_range=(ymin,ymax))
        self.p2.background_fill_color = self.bcolor
        self.p2.select(BoxSelectTool).select_every_mousemove = False
        self.p2.select(LassoSelectTool).select_every_mousemove = False

        self.r2 = self.p2.scatter(x=xlabel,y=ylabel, source=self.source, size=3, color=self.maincolor, alpha=0.6)
    
        self.l2 = self.p2.line(np.logspace(np.log10(lminlim),np.log10(maxlim),100),np.logspace(np.log10(lminlim),np.log10(maxlim),100),
                               color=self.outcolor)
        self.r2.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        self.panels.append(Panel(child=self.p2,title='semilogx'))


        # SEMILOGY TAB
        if ymin < 0:
            slymin = zp
        else:
            slymin = ymin
        self.p3 = figure(tools=TOOLS, plot_width=self.sqside, plot_height=self.sqside, 
                         min_border=10, min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left',
                         x_axis_label=xlabel,y_axis_label=ylabel,
                         x_axis_type='linear',y_axis_type='log',
                         x_range=(xmin,xmax),y_range=(slymin,ymax))
        self.p3.background_fill_color = self.bcolor
        self.p3.select(BoxSelectTool).select_every_mousemove = False
        self.p3.select(LassoSelectTool).select_every_mousemove = False

        self.r3 = self.p3.scatter(x=xlabel,y=ylabel, source=self.source, size=3, color=self.maincolor, alpha=0.6)
  
        self.l3 = self.p3.line(np.logspace(np.log10(lminlim),np.log10(maxlim),100),np.logspace(np.log10(lminlim),np.log10(maxlim),100),
                               color=self.outcolor)
        self.r3.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        self.panels.append(Panel(child=self.p3,title='semilogy'))


        # LOG PLOT
        self.p4 = figure(tools=TOOLS, plot_width=self.sqside, plot_height=self.sqside, 
                         min_border=10, min_border_left=50, toolbar_location="above", 
                         x_axis_location='below', y_axis_location='left',
                         x_axis_label=xlabel,y_axis_label=ylabel,
                         x_axis_type='log',y_axis_type='log',
                         x_range=(slxmin,xmax),y_range=(slymin,ymax))
        self.p4.background_fill_color = self.bcolor
        self.p4.select(BoxSelectTool).select_every_mousemove = False
        self.p4.select(LassoSelectTool).select_every_mousemove = False

        self.r4 = self.p4.scatter(x=xlabel,y=ylabel, source=self.source, size=3, color=self.maincolor, alpha=0.6)
        self.l4 = self.p4.line([lminlim,maxlim],[lminlim,maxlim],
                               color=self.outcolor)
        self.r4.nonselection_glyph = Circle(fill_color=self.unscolor, 
                                            fill_alpha=0.1, line_color=None)

        self.panels.append(Panel(child=self.p4,title='log'))

        self.ps = [self.p1,self.p2,self.p3,self.p4]
        self.rs = [self.r1,self.r2,self.r3,self.r4]
        self.ls = [self.l1,self.l2,self.l3,self.l4]

        # NEEDS CHECK FOR <= 0 POINTS

    def histograms(self,update=False):
        self.pt1 = prophist(self,'Efficiency',bins=np.linspace(0,1,20))
        self.pt1.plot_hist(x_range=(0,1),yscale='log')
        self.pb1 = prophist(self,'Completeness',bins=np.linspace(0,1,20))
        self.pb1.plot_hist(x_range=(0,1),yscale='log')
        self.pt2 = prophist(self,'Found Silhouette',bins=np.linspace(-1,1,40))
        self.pt2.plot_hist(x_range=(0,1),yscale='log')
        self.pb2 = prophist(self,'Matched Silhouette',bins=np.linspace(-1,1,40))
        self.pb2.plot_hist(x_range=(0,1),yscale='log',background=self.tsil)
        self.pt3 = prophist(self,'Found Size',bins= np.logspace(0,3,20))
        self.pt3.plot_hist(xscale='log',yscale='log',background=self.tsize)
        self.pb3 = prophist(self,'Matched Size',bins= np.logspace(0,3,20))
        self.pb3.plot_hist(xscale='log',yscale='log',background=self.tsize)
        self.proplist = [self.pt1,self.pb1,self.pt2,self.pb2,self.pt3,self.pb3]


    def buttons(self):
        files = np.array([g.split('_') for g in glob.glob('*.hdf5')])
        cases = [n.split('case')[1] for n in np.unique(files[:,0])]
        times = [n.split('.')[0] for n in np.unique(files[:,1])]
        self.selectcase = Select(title='case',value=str(self.case),options=cases)
        self.selecttime = Select(title='timestamp',value=str(self.timestamp),options=times)

        self.labels = list(self.datadict.keys())

        self.xradio = RadioButtonGroup(labels=self.labels, active=0,name='x-axis')
        self.yradio = RadioButtonGroup(labels=self.labels, active=1,name='y-axis')

        self.streps = list(np.unique(self.eps).astype(str))
        self.strmin = list(np.unique(self.min_samples).astype(str))
        self.selecteps = Select(title="eps value", value=str(self.epsval), 
                           options=self.streps)
        self.selectmin = Select(title="min_samples value", value=str(self.minval), 
                           options=self.strmin)

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
        print('Updating the histograms')
        inds = np.array(new['1d']['indices'])
        if len(inds) == 0 or len(inds) == self.numc:
            for i,prop in enumerate(self.proplist):
                prop.h1.data_source.data['top'] = 'zeros'
        else:
            neg_inds = np.ones_like(self.eff, dtype=np.bool)
            neg_inds[inds] = False
            for i,prop in enumerate(self.proplist):
                hist = (np.histogram(prop.arr[inds],bins=prop.edges)[0]).astype('float')
                prop.data['active'] = hist
                try:
                    print(prop.h1.data_source.data['active'])
                except:
                    pass
                prop.h1.data_source = ColumnDataSource(data=prop.data)
                print(prop.h1.data_source.data['active'])
                #prop.h1.data_source.data['top'] = 'active'
                prop.h1.glyph.top='active'

    def updateallx(self,new):
        newx = self.datadict[self.labels[new]]
        axlims,lineparams = findextremes(newx,self.r1.data_source.data[self.labels[self.yradio.active]],pad=self.pad)
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

        self.p1.x_range.start=xmin
        self.p1.x_range.end=xmax

        self.p2.x_range.start = slxmin
        self.p2.x_range.end = xmax

        self.p3.x_range.start = xmin
        self.p3.x_range.end = xmax

        self.p4.x_range.start = slxmin
        self.p4.x_range.end = xmax

        for r in self.rs:
            r.glyph.x = self.labels[new]

        self.l1.data_source.data['x'] = lineparams
        self.l1.data_source.data['y'] = lineparams

        self.l2.data_source.data['x'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)
        self.l2.data_source.data['y'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)

        self.l3.data_source.data['x'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)
        self.l3.data_source.data['y'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)

        self.l4.data_source.data['x'] = [lminlim,maxlim]
        self.l4.data_source.data['y'] = [lminlim,maxlim]
    
        for p in self.ps:
            p.xaxis.axis_label = self.labels[new]

    def updateally(self,new):
        newy = self.datadict[self.labels[new]]
        axlims,lineparams = findextremes(self.r1.data_source.data[self.labels[self.xradio.active]],newy,pad=self.pad)
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

        self.p1.y_range.start=ymin
        self.p1.y_range.end=ymax

        self.p2.y_range.start = ymin
        self.p2.y_range.end = ymax

        self.p3.y_range.start = slymin
        self.p3.y_range.end = ymax

        self.p4.y_range.start = slymin
        self.p4.y_range.end = ymax

        for r in self.rs:
            r.glyph.y = self.labels[new] 

        self.l1.data_source.data['x'] = lineparams
        self.l1.data_source.data['y'] = lineparams

        self.l2.data_source.data['x'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)
        self.l2.data_source.data['y'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)

        self.l3.data_source.data['x'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)
        self.l3.data_source.data['y'] = np.logspace(np.log10(lminlim),np.log10(maxlim),100)

        self.l4.data_source.data['x'] = [lminlim,maxlim]
        self.l4.data_source.data['y'] = [lminlim,maxlim]
    
        for p in self.ps:
            p.yaxis.axis_label = self.labels[new]

    def updatedata_eps(self,attr,old,new):
        self.read_run_data(ind=None, eps=float(new), min_samples=self.minval, neighbours=20)
        self.layout_plots(update=True)

starter = display_result(timestamp='2018-07-09.19.50.41.862297',pad=0.1)

# plot_eps = 0.5
# def updateeps(attr,old,new):
#     epsval = float(selecteps.value)
#     minval = float(selectmin.value)
#     runind = np.where((eps==epsval) & (min_samples==minval))[0]





# data.close()
#toggleline.on_click()
