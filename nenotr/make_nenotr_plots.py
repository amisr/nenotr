# -*- coding: utf-8 -*-
import os
import numpy as np
import tables
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from matplotlib import cm
import matplotlib.dates
from datetime import datetime


def make_grid_of_axes(nrows,ncols,dx=0.02,dy=0.05,figsz=(14,10)):

    figBG   = 'w'        # the figure background color
    axesBG  = '#f6f6f6'  # the axies background color

    if ncols==1 and nrows==1:
        POS=[0.1,0.1,1.0/(ncols+3.0)-dx,1.0/(nrows)-dy*5]
    elif ncols==1:
        POS=[0.1,0.65,1.0/(ncols+3.0)-dx,1.0/(nrows)-dy*2]
    elif nrows==1:
        POS=[0.1,0.1,1.0/(ncols+1)-dx,1.0/(nrows)-dy*5]
    elif ncols==-1 and nrows==-1:
        ncols=1; nrows=2; dx=0.1
        POS=[0.1,0.1,1.0-dx,1.0/(nrows)-dy*5]
    else:
        fy=1.0/(nrows)-dy*1.5
        POS=[0.1,1.0-fy-dy*1.5,1.0/(ncols+1)-dx,fy]

    figg=pyplot.figure(figsize=figsz, facecolor=figBG)
    ax=[]
    for aa in range(nrows):
        for bb in range(ncols):
            rect=[POS[0]+(POS[2]+dx)*bb,POS[1]-(POS[3]+dy)*aa,POS[2],POS[3]]
            ax.append(figg.add_axes(rect, facecolor=axesBG))

    return figg,ax


def timegaps(time,data,rngOpt=[]):

    if len(rngOpt)>0:
        doRng=1
        rng2=[]

    time2=[]
    if np.ndim(data)==3:
        concnan=np.zeros((1,data.shape[1],data.shape[2]),dtype=data.dtype)*np.nan
    elif np.ndim(data)==2:
        concnan=np.zeros((1,data.shape[1]),dtype=data.dtype)*np.nan
    data2=data.copy()
    dt=np.median(np.diff(np.mean(time,axis=1)))
    for aa in range(time.shape[0]-1):
        time2.append(time[aa,0])
        if ( (time[aa+1,1]-time[aa,0]) > (dt*2.0) ):
            time2.append(time[aa,1])
            #print datetime.datetime.utcfromtimestamp(time[aa,1])
            if np.ndim(data)==3:
                data2=np.concatenate((data2[0:len(time2)-1,:,:],concnan,data2[len(time2)-1:,:,:]),axis=0)
            elif np.ndim(data)==2:
                data2=np.concatenate((data2[0:len(time2)-1,:],concnan,data2[len(time2)-1:,:]),axis=0)

    time2.append(time[-1,0])
    time2.append(time[-1,1])

    return np.array(time2), data2


def pcolor_plot(x,y,data,cax,xlim,ylim,xl,yl,title,text,bmcodes,save_fig_name=None,max_beams=11,log=0):
    # Scale factor for number of x-tick markers when
    # more than 12 hours of data is being plotted.
    sc=1.0

    # Set some font sizes
    textsize=8.0
    labsize=12.0
    axlabsize=10.0

    # Label alignment adjustment values
    xxx=0.0
    yyy=0.0

    # Initialize some variables for figuring out how many groups of beams
    # we will need to plot
    num_beams = data.shape[1]
    remaining_beams_to_plot = num_beams
    start_beam = 0

    # How many plots do we need to make?
    num_of_figures = int(np.ceil(num_beams/float(max_beams)))

    # Colormap to use.
    cmap='jet'
    cmap_to_use = cm.get_cmap(cmap)
    cmap_to_use.set_bad('w',0)

    # Then set up some x-axis time formatting that is common to all of the
    # of the groups of beams we will plot.
    num_hours       = (xlim[-1] - xlim[0]) / 3600.0
    num_half_days   = num_hours / 12.0
    if num_half_days > 0.5:
        interval    = int(np.ceil(num_hours / 12.0))
        locator     = matplotlib.dates.HourLocator(interval=interval)
        formatter   = matplotlib.dates.DateFormatter("%H")
    elif num_half_days < 0.5:
        interval    = int(np.ceil(num_hours * 60.0 / 5.0 / sc))
        locator     = matplotlib.dates.MinuteLocator(interval=interval)
        formatter   = matplotlib.dates.DateFormatter("%H:%M")

    # Convert times to from epoch to numbers
    x = matplotlib.dates.epoch2num(x)
    xlim = [matplotlib.dates.epoch2num(xlim[0]),matplotlib.dates.epoch2num(xlim[-1])]

    # How many beams do we have left to plot?
    if remaining_beams_to_plot > max_beams:
        beams_to_plot = max_beams
    else:
        beams_to_plot = remaining_beams_to_plot

    # Now let's figure out the grid that we are going to make
    # by determining the number of rows and columns we need.
    num_axes_needed = beams_to_plot + 1
    if num_beams == 1:
        nrows = 1
        ncols = 2
    else:
        nrows = int(np.ceil(np.sqrt(num_axes_needed)))
        ncols = nrows
        if nrows * (nrows - 1) >= num_axes_needed:
            ncols = nrows - 1


    for fig_num in range(num_of_figures):
        # How many beams do we have left to plot?
        if remaining_beams_to_plot > max_beams:
            beams_to_plot = max_beams
        else:
            beams_to_plot = remaining_beams_to_plot
        remaining_beams_to_plot -= beams_to_plot

        # Now set up the figure and axes
        fig, ax = make_grid_of_axes(nrows,ncols)

        # And start plotting data, one beam at a time.
        for ii in range(beams_to_plot):
            iiB = start_beam + ii

            ax[ii].clear()

            data_plot = data[:,iiB,:]

            num_x, num_y    = data_plot.shape
            temp_y          = y[iiB,:]
            temp_y          = np.repeat(temp_y[np.newaxis,:],num_x,axis=0)
            temp_y_diff     = np.repeat(np.diff(temp_y[0,:])[np.newaxis,:],num_x,axis=0)
            y_diff          = np.zeros(temp_y.shape)
            y_diff[:,0:-1]  = temp_y_diff
            y_diff[:,-1]    = temp_y_diff[:,-1]

            # Construct the range array for plotting
            y_plot              = np.zeros((num_x+1,num_y+1))
            y_plot[0:-1,0:-1]   = temp_y - y_diff/2
            y_plot[0:-1,-1]     = temp_y[:,-1] + y_diff[:,-1]/2
            y_plot[-1,:]        = y_plot[-2,:]

            # Construct the time array for plotting
            x_plot = np.zeros((num_x+1,num_y+1))
            x_plot = np.repeat(x[:,np.newaxis],num_y+1,axis=1)

            # Use pcolormesh to plot the data.
            pc = ax[ii].pcolormesh(x_plot,y_plot,data_plot,vmin=cax[0],vmax=cax[1],cmap=cmap_to_use) #shading='interp',

            # Set axis formatting and labels
            ax[ii].xaxis.set_major_locator(locator)
            ax[ii].xaxis.set_major_formatter(formatter)
            ax[ii].set_xlim(xlim)
            ax[ii].set_ylim(ylim)
            if np.mod(ii,ncols)==0:
                ax[ii].set_ylabel(yl, fontsize=labsize)
            else:
                ax[ii].set_yticklabels([])

            ax[ii].tick_params(axis='both',labelsize=textsize)

            # Determine if we are on the last row of beams to be plotted, if so
            # then label the x-axis appropriately.
            if np.ceil((ii+1)/float(ncols)) >= np.ceil(beams_to_plot/float(ncols)):
                ax[ii].set_xlabel(xl, fontsize=labsize)

            tt=r"$%d \ (%.1f^o \ \rm{az} , \ %.1f^o \ \rm{el})$" % (iiB+1,bmcodes[iiB,1],bmcodes[iiB,2])
            ax[ii].set_title(tt, fontsize=axlabsize, horizontalalignment='center')
            if ii==0:
                ax[ii].text(xlim[0]+(xlim[1]-xlim[0])/2+xxx,(ylim[1]-ylim[0])*0.05*nrows+ylim[1]+yyy,title,fontsize=labsize, horizontalalignment='left')

        # Now we need to do the colorbar
        ii=ii+1
        # Scale the colorbar and change it's positions slightly.
        try:
            tmp=ax[ii].get_position()
            gp=np.array([tmp.xmin,tmp.ymin,tmp.width,tmp.height])
        except:
            gp=ax[ii].get_position()
        gp[1]=gp[1]+gp[3]
        gp[3]=gp[3]/6.
        gp[1]=gp[1]-gp[3]
        ax[ii].set_position(gp)
        if log:
            cl=pyplot.colorbar(pc,ax[ii],orientation='horizontal',format=pyplot.FormatStrFormatter('$10^{%.1f}$'))
        else:
            cl=pyplot.colorbar(pc,ax[ii],orientation='horizontal')
        ax[ii].yaxis.set_ticklabels([])
        ax[ii].tick_params(axis='y',labelsize=textsize)

        t = ax[ii].get_xticklabels()
        ax[ii].set_xticklabels(t,rotation=90)
        cl.set_label(text,fontsize=labsize*1.25)

        # Set any remaining axes invisible (ones where we didn't plot data)
        for jj in range(ii+1,len(ax)):
            ax[jj].set_visible(False)

        # Finally, save the figure if a name was provided.
        # Form the save name for the figure
        if save_fig_name is not None:
            if num_of_figures > 1:
                temp, extension = os.path.splitext(save_fig_name)
                output_name = temp + '_bmgrp-%d' % fig_num
                output_name += extension
            else:
                output_name = save_fig_name

            fig.savefig(output_name)

        # To help limit RAM usage, clear the figure from memory after
        # plotting and saving is done
        pyplot.close('all')

        # Increment the start_beam so we can keep track of which beam
        # is the next "first" one to plot.
        start_beam += beams_to_plot


def pcolor_plot_all(plot_info, data):

    # get required time information
    unix_time   = data['Time']['UnixTime']
    max_time    = plot_info['max_time']

    # Determine how many time groups of plots we should make
    total_time = (unix_time[-1,-1] - unix_time[0,0]) / 3600.0
    num_time_groups = np.ceil(total_time / max_time)

    # Check if the output path exists
    print(plot_info['plotsdir'])
    if not os.path.exists(plot_info['plotsdir']):
        raise IOError("Specified path: ''%s'' does not exist!" % plot_info['plotsdir'])


    print("There will be %d time groups of plots made." % num_time_groups)

    # First make the plots of the NeFromPower parameters
    altitude        = data['NeFromPower']['alt'] / 1000.0
    inds            = np.where(~np.isfinite(altitude))     # set all non finite values nan
    altitude[inds]  = np.nan

    nenotr = data['NeFromPower']['nenotr']
    inds = np.where(nenotr < 0)
    nenotr[inds] = np.nan
    nenotr = np.real(np.log10(nenotr))

    snr = data['NeFromPower']['snr']
    snr[inds] = np.nan
    snr = 10.0*np.real(np.log10(snr))

    # Determine the y-axis limits for the nonfitted data
    if len(plot_info['nonfitted_ylim'])==0:
        nonfitted_ylim=[np.nanmin(altitude), np.nanmax(altitude)]
    else:
        nonfitted_ylim = plot_info['nonfitted_ylim']

    start_ind=0;
    for time_ind in range(int(num_time_groups)):
        end_ind = np.where(unix_time[:,-1] <= (unix_time[start_ind,0] + max_time * 3600.0))[0]
        end_ind = end_ind[-1]
        tlim = [start_ind,end_ind]
        start_ind = end_ind+1

        if num_time_groups>1:
            txtra='_day' + str(time_ind)
        else:
            txtra=''

        # Figure out the time text for the title of the plots
        title =  "%d-%d-%d "    % (data['Time']['Month'][tlim[0],0],data['Time']['Day'][tlim[0],0],data['Time']['Year'][tlim[0],0])
        title += "%.3f UT "     %  data['Time']['dtime'][tlim[0],0]
        title += "- %d-%d-%d "  % (data['Time']['Month'][tlim[-1],1],data['Time']['Day'][tlim[-1],1],data['Time']['Year'][tlim[-1],1])
        title += "%.3f UT"      %  data['Time']['dtime'][tlim[-1],1]

        # Set up the x-axis stuff
        xlim                = [unix_time[tlim[0],0], unix_time[tlim[1],1]]
        trimmed_unix_time   = unix_time[tlim[0]:(tlim[1]+1)]


        # Plot the uncorrected density nenotr (Te/Ti)
        clim = plot_info['clims'][0]
        txt  = r'$\rm{Ne_NoTr} \ (\rm{m}^{-3})$'

        plot_times, plot_datas = timegaps(trimmed_unix_time,nenotr[tlim[0]:tlim[1]+1])
        plot_datas = np.ma.masked_where(np.isnan(plot_datas),plot_datas)

        if plot_info['saveplots']==1:
            oname = title + '_Ne_NoTr' + txtra + '.png'
            output_fig_name = os.path.join(plot_info['plotsdir'],oname)
        else:
            output_fig_name = None

        pcolor_plot(plot_times,altitude,plot_datas,clim,xlim,nonfitted_ylim,'Time (UT)','Altitude (km)',
                    title,txt,data['BeamCodes'],save_fig_name=output_fig_name,max_beams=plot_info['max_beams'],log=1)

        # Plot the SNR now.
        clim = [-20.0,10.0]
        txt  = r'$\rm{SNR} \ (\rm{dB})$'

        plot_times, plot_datas = timegaps(trimmed_unix_time,snr[tlim[0]:tlim[1]+1])
        plot_datas = np.ma.masked_where(np.isnan(plot_datas),plot_datas)

        if plot_info['saveplots']==1:
            oname = title + '_SNR' + txtra + '.png'
            output_fig_name = os.path.join(plot_info['plotsdir'],oname)
        else:
            output_fig_name = None

        pcolor_plot(plot_times,altitude,plot_datas,clim,xlim,nonfitted_ylim,'Time (UT)','Altitude (km)',
                    title,txt,data['BeamCodes'],save_fig_name=output_fig_name,max_beams=plot_info['max_beams'],log=0)



def has_fitted_params(fname):

    with tables.open_file(fname) as h5:
        try:
            h5.get_node('/FittedParams')
            fitted_params = True
        except tables.NoSuchNodeError:
            fitted_params = False

    return fitted_params


def replot_pcolor_all(fname,saveplots=0,opath='.',clims=[[10,12],[0,1500],[0,3000],[0,4],[-500,500]],nonfitted_ylim=[],fitted_ylim=[],max_time=24.0,max_beams=11):
    # Read the entire data file. Really we only need, Ne, Te, Ti, Tr, Vlos, Frac, SNR, nuin, Ne_NoTrer_NoTr
    # Also we want to plot the errors in eNe, eTe, eTi, eVlos
    #max_time sets the maximum number of hours of data to plot in one plot
    #max_beams sets the number of beams to plot in a plot

    # Initialize a plotting information dictionary and data dictionary
    plot_info = dict()
    data = dict()
    data['NeFromPower'] = dict()
    data['Time'] = dict()
    data['FittedParams'] = dict()

    # Copy the plotting configuration information
    plot_info['saveplots']      = saveplots
    plot_info['plotsdir']       = opath
    plot_info['clims']          = clims
    plot_info['nonfitted_ylim'] = nonfitted_ylim
    plot_info['fitted_ylim']    = fitted_ylim
    plot_info['max_time']       = max_time
    plot_info['max_beams']      = max_beams


    # Now we'll read in the data that we need
    with tables.open_file(fname) as h5:
        # Ne from Power data
        data['NeFromPower']['snr']      = h5.root.NeFromPower.SNR.read()
        data['NeFromPower']['nenotr']  = h5.root.NeFromPower.Ne_NoTr.read()
        data['NeFromPower']['rng']      = h5.root.NeFromPower.Range.read()
        data['NeFromPower']['alt']      = h5.root.NeFromPower.Altitude.read()

        # Get the Beam Codes
        data['BeamCodes'] = h5.root.BeamCodes.read()

        # Finally, grab the time and site information
        data['Time']['UnixTime']    = h5.root.Time.UnixTime.read()
        data['Time']['Day']         = h5.root.Time.Day.read()
        data['Time']['Month']       = h5.root.Time.Month.read()
        data['Time']['Year']        = h5.root.Time.Year.read()

    dutcfts = datetime.utcfromtimestamp
    dts = np.array([[dutcfts(x[0]),dutcfts(x[1])] for x in data['Time']['UnixTime']])
    def dtime(dt):
        tmp = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        return tmp
    data['Time']['dtime'] = np.array([[dtime(x[0]),dtime(x[1])] for x in dts])

    pcolor_plot_all(plot_info, data)

    return


def usage():
    print("usage: ", sys.argv[0])
    print("\t DATAFILE: hdf5 file of fitted data [REQUIRED]")
    print("\t PLOTDIR: directory to place plots in [OPTIONAL]")

    sys.exit(2)

if __name__ == "__main__":

    import sys

    # Parse input
    data_file = sys.argv[1]

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        usage()
    elif len(sys.argv) == 2:
        plots_dir = './plots'
    else:
        plots_dir = sys.argv[2]

    # Check if the output plot directory exists
    if not os.path.exists(plots_dir):
        # If it doesn't, make it.
        try:
            os.mkdir(plots_dir)
        except OSError:
            print("Problem making the output plotting directory, exiting...")
            sys.exit(1)

    # Make the plots
    now = datetime.now()
    replot_pcolor_all(data_file,saveplots=1,opath=plots_dir) #,clims=clims)
    print('It took %d seconds to plot the data.' % (datetime.now()-now).total_seconds())
