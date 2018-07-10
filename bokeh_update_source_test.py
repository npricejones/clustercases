from bokeh.plotting import figure, output_file, curdoc
from bokeh.models import CustomJS, ColumnDataSource, Select
from bokeh.layouts import gridplot
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.layouts import widgetbox,column

from random import random

key_list = list('ABCDEFGHIJKLMNOP')

DATA1 = {key:[random() for i in range(10)] for key in key_list}
DATA2 = {key:[random() for i in range(15)] for key in key_list}
DATA1['xaxis'] = range(10)
DATA2['xaxis'] = range(15)

source1 = ColumnDataSource(data=DATA1)
source2 = ColumnDataSource(data=DATA2)

fill_source = ColumnDataSource(data=DATA1)

fig = figure()

for key in key_list:
    fig.circle(x='xaxis',y=key,source=fill_source)

select = Select(options=['source1','source2'],value='source1')

codes = """
var f = cb_obj.value;
var sdata = source.data;
var data1 = source1.data;
var data2 = source2.data;

if (f == "source1") {
for (key in data1) {
    sdata[key] = [];
    for (i=0;i<data1[key].length;i++){
    sdata[key].push(data1[key][i]);
    }
}
} else {
for (key in data2) {
    sdata[key] = [];
    for (i=0;i<data2[key].length;i++){
    sdata[key].push(data2[key][i]);
    }
}
};

source.change.emit();
"""
select.callback = CustomJS(args=dict(source=fill_source,source1=source1,source2=source2),code=codes)

curdoc().add_root(column(widgetbox(select), fig, width=1100))
