
import os
import numpy as np
import json
import argparse
import sys
from functools import partial

import bokeh.plotting as bp
from bokeh.models import RadioButtonGroup
from bokeh.models.layouts import Column, Row
from bokeh.models.widgets import Div, Select
from bokeh.layouts import widgetbox, Spacer

FIXEDIDX = 0

class Errorsviewer:
    def __init__(self, resdir):
        self.config = {}
        self.fig = None
        self.contr = {}
        self.currparams = {}
        self.var = 'sensorsnum'
        self.resdir = resdir
        self.cols = [ '#b3de69', '#bebada', '#fb8072', '#80b1d3', 
            '#fdb462', '#8dd3c7', '#fccde5', '#d9d9d9', 
            '#bc80bd', '#ffffb3']

        if os.path.exists(os.path.join(resdir, 'config.json')):
            with open(os.path.join(resdir, 'config.json'), 'r') as fh:
                self.config = json.load(fh)
        else:
            for ff in os.listdir(resdir):
                if not ff.endswith(".json"): continue
                config = json.load(open(os.path.join(resdir, ff)))

                if self.config:
                    for k, v in config.items():
                        if type(v) != list: continue
                        self.config[k] += v
                else:
                    self.config = config

            for k in self.config.keys():
                if type(self.config[k]) == list:
                    self.config[k] = sorted(list(set(self.config[k])))

        self.create_gui()
        self.scaling = 'linear'
        self.add_controllers()
        self.update_plot()
        self.blocked = False
        self.scalingwidget = []
    
    ##########################################################
    def create_gui(self):
        self.guirow = Row(widgetbox(), widgetbox())
        bp.curdoc().add_root(self.guirow)
        bp.curdoc().title = 'Simulation results of {}'.format(self.resdir)

    ##########################################################
    def add_controllers(self):
        titles = {'npeople': 'Number of people',
                  'sensorinterval': 'Interval',
                  'sensortpr': 'True positive rate',
                  'sensorexfp': 'Expected num of FP',
                  'sensorrange': 'Sensor range',
                  'sensorsnum': 'Fleet size',
                  'sensorspeed': 'Fleet speed'}
        idx0 = 1

        stacked = []

        #Create a controller for the scaling
        def on_scaling_changed(_, _old, _new):
            self.scaling = _new
            self.update_plot()

        self.scalingwidget = Select(title='Scaling', value='linear',
                              options=['linear', 'log'])
        self.scalingwidget.on_change('value', on_scaling_changed)
        stacked.append(self.scalingwidget)
        # Create a controller for each param
        for k in titles.keys():
            def on_radio_changed(attr, old, new, kk):
                if self.blocked: return
                newvalue = self.config[kk][new-1] if new != FIXEDIDX else -1
                print('Changed ' + str(kk) + ' to ' + str(newvalue))
                self.currparams[kk] = newvalue

                if new == FIXEDIDX:
                    self.blocked = True
                    for param in self.contr.keys():
                        if param == kk or self.contr[param].active != FIXEDIDX:
                            continue

                        self.contr[param].active = 1
                        self.currparams[param] = self.config[param][0]
                    self.var = kk

                varyingparam = False
                for kkk, vvv in self.currparams.items():
                    if vvv == -1:
                        varyingparam = True
                        break
                if not varyingparam: self.var = None
                self.update_plot()
                if self.blocked: self.blocked = False

            my_radio_changed = partial(on_radio_changed, kk=k)
            params = ['varying'] + list(map(str, self.config[k]))

            buttonisactive = FIXEDIDX if self.var == k else idx0
            self.contr[k] = RadioButtonGroup(labels=params, active=buttonisactive)

            self.contr[k].on_change('active', my_radio_changed)
            self.currparams[k] = params[idx0]

            r = Row(widgetbox(Div(text='{}:'.format(titles[k]))),
                    self.contr[k])
            stacked.append(r)


        self.currparams[self.var] = -1
        adjwidget = Column(*stacked)
        self.guirow.children[0] = adjwidget

    ##########################################################
    def update_plot(self):
        #attr = 'fleetsize'
        attr = self.var

        titletxt = 'Densities error '
        if attr: titletxt += 'according to {}'.format(attr)
        yaxistitle = 'Error'
        if self.scaling == 'log':
            yaxistitle = 'Log-' + yaxistitle

        self.fig = bp.figure(plot_width=800, plot_height=600,
                             x_axis_label='Ticks',
                             y_axis_label=yaxistitle,
                             title=titletxt)
        nrepeats = int(self.config['nrepeats'])
        nticks = int(self.config['nticks'])

        deltax = 10

        if not self.var:
            p = self.currparams
            err = np.ndarray((nticks, nrepeats))

            acc = 0
            for r in range(nrepeats):
                filename = 'npeople{}_sensorsnum{}_sensorrange{}_' \
                    'sensorinterval{}_sensortpr{}_sensorexfp{}_' \
                    'sensorspeed{}_repeat{}.npy'. \
                    format(p['npeople'], p['sensorsnum'], p['sensorrange'],
                           p['sensorinterval'], p['sensortpr'],
                           p['sensorexfp'], p['sensorspeed'], r)
                fullfile = os.path.join(self.resdir, filename)
                if not os.path.exists(fullfile): continue
                aux = np.load(fullfile)
                err[:, acc] = aux[:, 2]
                acc += 1

            err = err[:, :acc-1]
            _means = np.mean(err, axis=1)

            if self.scaling == 'log': _means = np.log(_means)

            _stds = np.std(err, axis=1)
            if attr:
                legendtxt = '{} {} ({} runs)'. \
                    format(attr, self.currparams[attr], acc)
            else:
                legendtxt = ''

            self.plot_errorbar(range(0, _means.shape[0], deltax),
                               _means[0::deltax], _stds[0::deltax],
                               legendtxt, self.cols[0])
        else:
            p = self.currparams.copy()
            i = 0
            for v in self.config[self.var]:
                p[self.var] = v
                err = np.ndarray((nticks, nrepeats))

                acc = 0
                for r in range(nrepeats):
                    filename = 'npeople{}_sensorsnum{}_sensorrange{}_' \
                        'sensorinterval{}_sensortpr{}_sensorexfp{}_' \
                        'sensorspeed{}_repeat{}.npy'. \
                        format(p['npeople'], p['sensorsnum'], p['sensorrange'],
                               p['sensorinterval'], p['sensortpr'],
                               p['sensorexfp'], p['sensorspeed'], r)
                    fullfile = os.path.join(self.resdir, filename)
                    if not os.path.exists(fullfile): continue
                    aux = np.load(fullfile)
                    err[:, acc] = aux[:, 2]
                    acc += 1

                if acc == 0: continue
                err = err[:, :acc-1]

                _means = np.mean(err, axis=1)
                if self.scaling == 'log': _means = np.log(_means)
                _stds = np.std(err, axis=1)
                legendtxt = '{} {} ({} runs)'.format(attr, v, acc)
                _ticks = range(0, _means.shape[0], deltax)

                #_xaxis = _ticks
                _xaxis = np.array(_ticks)  * int(p['sensorsnum']) * int(p['sensorinterval'])

                self.plot_errorbar(_xaxis,
                                   _means[0::deltax], _stds[0::deltax], legendtxt,
                                   self.cols[i])
                i += 1

        self.guirow.children[1] = self.fig
    ##########################################################
    def plot_errorbar(self, x, y, errors, datatitle='', col='gray'):
        self.figline = self.fig.line(x, y, line_width=2, color=col,
                                     legend=datatitle)

        xflipped = np.flip(x, 0)
        yerrorupper = y + errors
        yerrorlower = np.flip(y, 0) - np.flip(errors, 0)

        yerrorbounds = np.append(yerrorupper, yerrorlower)
        self.fig.patch(x=np.append(x,xflipped), y=yerrorbounds, color=col, alpha=0.1)

##########################################################
# Do not put if__name__ main because it's called by bokeh

if len(sys.argv) == 2:
    resdir = sys.argv[1]
    viewer = Errorsviewer(resdir)
else:
    print('##########################################################')
    print('##########################################################')
    print('Not running!')
    print('Usage: $ bokeh serve --show src/errorsview.py --args <results-dir>')
    print('##########################################################')
    print('##########################################################')

