# -*- coding: utf-8 -*-
# Created: 4 May 2022
# Author: Pablo M. Reyes

import os
import numpy as np
import tables
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates

def do_plots(fname, plotfolder):
    print(f"reading {fname}")
    with tables.open_file(fname) as fp:
        BeamCodes = fp.root.BeamCodes.read()
        UnixTime = fp.root.Time.UnixTime.read()
        Altitude = fp.root.NeFromPower.Altitude.read()
        Ne_NoTr = fp.root.NeFromPower.Ne_NoTr.read()
    dts = np.array(
        [datetime.datetime.utcfromtimestamp(x) for x in UnixTime.mean(axis=-1)])

    bmi = 0
    vmin,vmax = 1e10,1e12
    hmin = 40
    hmax = Altitude.max()/1e3
    fontsize = plt.rcParams['font.size']*0.9
    nbeams = BeamCodes.shape[0]
    fig,axs = plt.subplots(figsize=(6,5),nrows =nbeams, dpi=130, gridspec_kw=dict(
                            hspace=0.3, top=0.9))
    for bmi,ax in enumerate(axs):

        pcm = ax.pcolormesh(dts,Altitude[bmi,:]/1e3,
                            Ne_NoTr[:,bmi,:].T,
                            norm=mpl.colors.LogNorm(vmin,vmax),
                           cmap = mpl.cm.jet)

        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        minorlocator = mdates.AutoDateLocator(minticks=9, maxticks=21)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_minor_locator(minorlocator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_ylim(hmin,hmax)
        if bmi < nbeams -1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (UT)')
        bcode,az,el,ksys = BeamCodes[bmi,:]
        ax.set_title(f"bm{bmi}, bcode:{int(bcode)}, ({az}\u00b0 az, {el}\u00b0 el)",
                     fontsize=fontsize, y=0.92)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=25))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=5))
        for l in ax.get_yticklabels() + ax.get_xticklabels():
            l.set_fontsize(fontsize*0.9)

    ax0 = fig.add_subplot(111, frame_on=False)   # creating a single axes
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_ylabel("Altitude (km)", labelpad=30)

    cb = fig.colorbar(pcm, ax = list(axs.ravel()), shrink=0.75, pad=0.02);
    cb.ax.set_ylabel(r'Ne_NoTr (m$^{-3}$)');
    exps = np.arange(10,12+0.25,0.25)
    cb.ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(10**exps))
    cb.ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator([]))
    #cb.ax.set_yticks(10**exps)
    cb.ax.set_yticklabels([r'10$^{%.2f}$'%x for x in exps])

    dt0 = dts[0]
    dt1 = dts[-1]
    dt0hr = dt0.hour+dt0.minute/60 + dt0.second/3600 + dt0.microsecond/3600/1e6
    dt0str = dt0.strftime(f'%m-%d-%Y {dt0hr:.3f} UT')
    dt1hr = dt1.hour+dt1.minute/60 + dt1.second/3600 + dt1.microsecond/3600/1e6
    dt1str = dt1.strftime(f'%m-%d-%Y {dt1hr:.3f} UT')
    interval = dt0str + ' - ' + dt1str


    fig.suptitle(interval, fontsize=fontsize);

    figname = os.path.join(plotfolder,f'{interval}_Ne_NoTr.png')
    print(f"saving {figname}")
    fig.savefig(figname, bbox_inches='tight', 
                   transparent=True,
                   pad_inches=0.1)
