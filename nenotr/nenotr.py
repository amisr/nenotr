# -*- coding: utf-8 -*-
# Created: 4 July 2018
# Author: Ashton S. Reimer

import re
import os
import glob
import tables
import numpy as np
from datetime import datetime
import tempfile
import configparser as ConfigParser

from .repack import Repackh5
from .datahandler import DataHandler

# output file definition
h5paths = [['/ProcessingParams','Experiment Parameters'],
           ['/NeFromPower','Electron density From Power'],
           ['/Site','Site Parameters'],
           ['/Time','Time Information'],
           #['Antenna','/Antenna','Antenna Motion Parameters'],
          ]

h5attribs = {'/BeamCodes' : [('TITLE','BeamCodes'),('Description','Beamcode array'),('Size','Nbeams x 4 (Beamcode, Azimuth (degrees), Elevation (degrees), System constant (m^5/s)')],
             '/NeFromPower/Altitude' : [('TITLE','Altitude'),('Unit','Meters'),('Description','Altitude assuming local flat Earth.')],
             '/NeFromPower/Range' : [('TITLE','Range'),('Unit','Meters')],
             '/NeFromPower/Ne_NoTr' : [('TITLE','Raw Electron Density'),('Description','Electron density from power. May contain range aliased F region and/or other range dependent noise sources.'),('Unit','m^{-3}'),('Size','Nrecords x Nbeams x Nranges')],
             '/NeFromPower/errNe_NoTr' : [('TITLE','Error in Raw Electron Density'),('Unit','m^{-3}'),('Size','Nrecords x Nbeams x Nranges')],
             '/NeFromPower/SNR' : [('TITLE','Signal to Noise Ratio'),('Description','SNR from power'),('Size','Nrecords x Nbeams x Nranges')],
             '/NeFromPower/errSNR' : [('TITLE','Error in Signal to Noise Ratio'),('Size','Nrecords x Nbeams x Nranges')],
             '/ProcessingParams/ProcessingTimeStamp' : [('TITLE','Processing Time Stamp')],
             '/ProcessingParams/BaudLength' : [('TITLE','Baud Length'),('Unit','Seconds')],
             '/ProcessingParams/PulseLength' : [('TITLE','Pulse Length'),('Unit','Seconds')],
             '/ProcessingParams/RxFrequency' : [('TITLE','Rx Frequency'),('Description','Receive frequency'),('Unit','Hertz')],
             '/ProcessingParams/TxFrequency' : [('TITLE','Tx Frequency'),('Description','Transmit frequency'),('Unit','Hertz')],
             '/ProcessingParams/TxPower' : [('TITLE','Tx Power'),('Unit','Watts'),('Description','Average transmit power over integration'),('Size','Nrecords')],
             '/ProcessingParams/AeuRx' : [('TITLE','Rx AEUs'),('Description','Number of AEUs on receive'),('Size','Nrecords')],
             '/ProcessingParams/AeuTx' : [('TITLE','Tx AEUs'),('Description','Number of AEUs on transmit'),('Size','Nrecords')],
             '/ProcessingParams/AeuTotal' : [('TITLE','Total AEUs'),('Description','Total number of system AEUs'),('Size','Nrecords')],
             '/Site/Altitude' : [('TITLE','Altitude'),('Description','Altitude of site'),('Unit','Meters')],
             '/Site/Code' : [('TITLE','Site Code')],
             '/Site/Latitude' : [('TITLE','Latitude'),('Description','Latitude of site'),('Unit','Degrees North')],
             '/Site/Longitude' : [('TITLE','Longitude'),('Description','Longitude of site'),('Unit','Degrees East')],
             '/Site/Name' : [('TITLE','Name'),('Description','Site Name')],
             '/Time/Day' : [('TITLE','Day of Month'),('Size','Nrecords x 2 (Start and end of integration')],
             '/Time/Month' : [('TITLE','Month'),('Size','Nrecords x 2 (Start and end of integration')],
             '/Time/Year' : [('TITLE','Year'),('Size','Nrecords x 2 (Start and end of integration')],
             '/Time/doy' : [('TITLE','Day of Year'),('Size','Nrecords x 2 (Start and end of integration')],
             '/Time/UnixTime' : [('TITLE','Unix Time'),('Size','Nrecords x 2 (Start and end of integration'),('Unit','Seconds')],
             # '/Antenna/AvgAzimuth' : [('TITLE','Average Azimuth Angle'),('Description','Average azimuth angle over integration'),('Size','Nrecords'),('Unit','Degrees')],
             # '/Antenna/AvgElevation' : [('TITLE','Average Elevation Angle'),('Description','Average elevation angle over integration'),('Size','Nrecords'),('Unit','Degrees')],
             # '/Antenna/Azimuth' : [('TITLE','Azimuth Angle'),('Description','Azimuth angle range over integration'),('Size','Nrecords x 2'),('Unit','Degrees')],
             # '/Antenna/Elevation' : [('TITLE','Elevation Angle'),('Description','Elevation angle range over integration'),('Size','Nrecords x 2'),('Unit','Degrees')],
             # '/Antenna/Event' : [('TITLE','Event'),('Description','Antenna event over integration'),('Size','Nrecords')],
             # '/Antenna/Mode' : [('TITLE','Mode'),('Description','Antenna mode over integration'),('Size','Nrecords')],
             }



class CalcNeNoTr(object):
    def __init__(self,configfile):

        # read the config file
        print("Reading configuration file...")
        self.configfile = configfile
        self.config = self.parse_config()

        # find files using config information
        print("Finding files...")
        self.filelists = self.find_files()

        # initialize the data handlers
        print("Checking available data...")
        self.num_freqs = len(self.filelists)
        self.datahandlers = [DataHandler(self.filelists[i]) for i in range(self.num_freqs)]

        # get experiment mode name
        self.mode_name = self.get_mode_name()
        print("Experiment mode is %s" % self.mode_name)

        # if a ksys file was provided, try to load it.
        print("Checking ksys file...")
        if self.config['input'].get('ksys_file', None) is None:
            self.ksys = self.__load_ksys_from_data()
            self.calibrated = False
            print("No file provided. Ksys loaded from data.")
        else:
            self.ksys = self.__load_ksys_from_file()
            self.calibrated = True
            print("Ksys file loaded.")

        # find all unique times from the data handlers and then
        # determine the integration periods to calculate Ne_NoTr for
        print("Calculating integration periods...")
        self.times = self.get_unique_times()
        self.integration_periods = self.get_integration_periods()

        # if no ambiguity function file is in the config, check if one is in 
        # the input files
        if self.config['input'].get('amb_path',None) is None:
            self.ambiguity = self.__load_amb_from_data()                # TODO! TEST THIS ON MSWINDS26.v03
        else:
            self.ambiguity = self.__load_amb_from_file()

        # make sure output directory is available and if not create it
        print("Validating output directory...")
        output_dir = self.config['output']['output_path']
        if not os.path.exists(output_dir):
            print("    Output directory doesn't exist!")
            print("    Attempting to create one...")
            os.makedirs(output_dir)


    def __load_ksys_from_data(self):
        fname = self.datahandlers[0].filelist[0]
        
        with tables.open_file(fname,'r') as h5:
            beamcodemap = h5.root.Setup.BeamcodeMap.read()

        return np.array(beamcodemap)


    def __load_ksys_from_file(self):
        return np.loadtxt(self.config['input']['ksys_file'])



    def __load_amb_from_data(self):
        fname = self.datahandlers[0].filelist[0]
        amb_path = "%s/Data/" % self.datahandlers[0].nenotr_mode
        
        with tables.open_file(fname,'r') as h5:
            node = h5.get_node(amb_path)
            bandwidth = node.Ambiguity.Bandwidth.read()
            wlag =node.Ambiguity.Wlag.read()

        return {'bandwidth': bandwidth, 'wlag': wlag}


    def __load_amb_from_file(self):
        fname = self.config['input']['amb_path']
        with tables.open_file(fname,'r') as h5:
            bandwidth = h5.root.Bandwidth.read()
            wlag = h5.root.Wlag.read()

        return {'bandwidth': bandwidth, 'wlag': wlag}


    def parse_config(self):
        required_sections_options = {'default': {'integ': str},
                                     'input': {'file_paths': str},
                                     'output': {'output_name':str,
                                                'output_path': str,
                                               },
                                     'nenotr_options': {'mean_or_median': str,
                                                       'recs2integrate': int},
                                    }

        optional_sections_options = {'input': {'amb_path': str,
                                               'ksys_file': str
                                              },
                                    }

        # read the config file and convert to dictionary
        parser = ConfigParser.ConfigParser()
        parser.read(self.configfile)
        parsed_config = self.__config_to_dict_helper(parser)

        # check the config file to make sure we have all required information
        for section in required_sections_options.keys():
            if parsed_config.get(section,None) is None:
                msg = 'Required section: "%s" is missing from config.' % section
                raise AttributeError(msg)
            for option in required_sections_options[section].keys():
                if parsed_config[section].get(option,None) is None:
                    msg = 'Required option: "%s" is missing' % option
                    msg += ' from the "%s" section in the config.' % section
                    raise AttributeError(msg)

                # convert the input config data to the required format
                type_func = required_sections_options[section][option]
                converted = type_func(parsed_config[section][option])
                parsed_config[section][option] = converted

        # make sure optional options are formatted as required
        for section in optional_sections_options.keys():
            for option in optional_sections_options[section].keys():
                # convert the input config data to the required format
                type_func = optional_sections_options[section][option]
                try:
                    converted = type_func(parsed_config[section][option])
                    parsed_config[section][option] = converted
                except KeyError:
                    pass

        return parsed_config


    def get_unique_times(self):
        all_times = list()
        for i in range(self.num_freqs):
            all_times.extend(list(self.datahandlers[i].times))
        all_times = [tuple(x) for x in all_times]
        unique_times = np.array(sorted(list(set(list(all_times)))))

        # now detect time pairs that have 0 difference in start or end time
        # sometimes raw files don't have exactly the same time windows...
        if self.num_freqs > 1:
            diffs = np.diff(unique_times,axis=0)   # diff the start and end times
            diffs = np.array([[x[0].total_seconds(),x[1].total_seconds()] for x in diffs])
            inds = np.where(~((diffs[:,0] == 0) | (diffs[:,1] == 0)))[0]  # exclude times where start or end diffs are 0
            unique_times = unique_times[inds,:]

        return unique_times


    def get_integration_periods(self):
        integration_periods = list()
        start_time = None
        integration_time = self.config['nenotr_options']['recs2integrate']
        num_times = len(self.times)
        for i,time_pair in enumerate(self.times):
            temp_start_time, temp_end_time = time_pair
            if start_time is None:
                start_time = temp_start_time
            time_diff = (temp_end_time - start_time).total_seconds()

            if time_diff >= integration_time:
                integration_periods.append([start_time,temp_end_time])
                start_time = None
                continue

            # Add an integration period for when we are at the end of the files
            # but we haven't reached the requested integration time
            if (i == num_times -1):
                integration_periods.append([start_time,temp_end_time])

        return np.array(integration_periods)


    def find_files(self):
        # we need to find all files that match the search strings
        # and check every input file path for them
        search_paths_by_freq = self.config['input']['file_paths'].split(',')
        num_freqs = len(search_paths_by_freq)

        filelists = [[] for x in range(num_freqs)]
        for i in range(num_freqs):
            paths = search_paths_by_freq[i].split(':')
            temp = list()
            for path in paths:
                files_found = glob.glob(path)
                num_files_found = len(files_found)
                if num_files_found == 0:
                    print('No files matching "%s"' % str(path))
                temp.extend(glob.glob(path))

            filelists[i].extend(sorted(temp))

        # Now trim frequencies with no files in them
        i = 0
        while i < len(filelists):
            if len(filelists[i]) == 0:
                filelists.pop(i)
            else:
                i += 1
        num_freqs = len(filelists)

        # calculate the total number of files
        num_files = 0
        for i in range(len(filelists)):
            num_files += len(filelists[i])

        if num_files == 0:
            raise Exception('No files found!')

        filestr = "files" if num_files > 1 else "file"
        freqstr = "frequencies" if num_freqs > 1 else "frequency"
        print("Found %s %s for %s %s." % (num_files,filestr,num_freqs,freqstr))

        return filelists


    def get_mode_name(self):
        # could add a check to make sure all files are from the same experiment mode?
        fname = self.datahandlers[0].filelist[0]
        print("fname in get_mode_name",fname)
        try:
            with tables.open_file(fname) as h5file:
                expname = h5file.get_node('/Setup/Experimentfile').read()
            print("type(expname)",type(expname))
            if type(expname)==np.ndarray:
                expname=expname[0]
            print("type(expname)",type(expname))
            expname=expname.splitlines()[1].split(b'=')[1] # byte-like obj. python3
        except Exception as e:
            print("Could not determine mode name because: %s" % (str(e)))
            print("Defaulting to 'unknown'.")
            try:
                expname=bytes('unknown')         #python2
            except:
                expname=bytes('unknown','utf-8') #python3

        return expname


    @staticmethod
    def __config_to_dict_helper(configparserclass):
        # based on https://gist.github.com/amitsaha/3065184
        # converts a config parser object to a dictionary
        config = dict()
        defaults = configparserclass.defaults()
        sections = configparserclass.sections()

        temp = dict()
        for key in defaults:
            temp[key] = defaults[key]
        config['default'] = temp
        default_options = temp.keys()

        for section in sections:
            opts = configparserclass.options(section)
            options = [x for x in opts if not x in default_options]
            temp = dict()
            for option in options:
                temp[option] = configparserclass.get(section,option)
            config[section.lower()] = temp

        return config


    # Take all data from all frequencies for each integrated time step,
    # pass this to a function that formats things correctly and then 
    # pass the output of that into the Ne_NoTr calculation code
    # This function gets averaged quantities per frequency from another
    # function. Then it combines everything, calculates density, and 
    # then calculates the variances in everything.
    # Code doesn't write anything to disk, uses RAM.
    def calculate_nenotr(self):
        if self.config['nenotr_options']['mean_or_median'] == 'mean':
            func = np.nanmean
        if self.config['nenotr_options']['mean_or_median'] == 'median':
            func = np.nanmedian        

        array_keys = ['signal','snr','calsignal','noise','uncal_signal',
                      'power','calwithnoise','noisepower']
        sample_keys = ['noisepowersamples','powsamples','calsamples']
        nenotr = dict()
        nenotr['snr'] = None
        signal_ranges = None
        num_integrations = len(self.integration_periods)

        for i,integration_period in enumerate(self.integration_periods):
            print("Integration period %s/%s" % (str(i+1),str(num_integrations)))
            for j,datahandler in enumerate(self.datahandlers):
                data, _ = datahandler.get_records(integration_period[0],integration_period[1])
                data['ambiguity'] = self.ambiguity

                output = self.get_signal_and_snr(data,self.config)

                if nenotr['snr'] is None:
                    num_beams, num_ranges = output['snr'].shape
                    
                    for key in array_keys:
                        nenotr[key] = np.zeros((num_integrations,num_beams,num_ranges,self.num_freqs))
                    for key in sample_keys:
                        nenotr[key] = np.zeros((num_integrations,num_beams,num_ranges,self.num_freqs))

                    nenotr['txpower'] = np.zeros((num_integrations,))
                    nenotr['aeurx'] = np.zeros((num_integrations,))
                    nenotr['aeutx'] = np.zeros((num_integrations,))
                    nenotr['aeutotal'] = np.zeros((num_integrations,))
                    nenotr['txfreq'] = np.zeros((num_integrations,self.num_freqs))
                    nenotr['rxfreq'] = np.zeros((num_integrations,self.num_freqs))

                    # quantities that do not vary with time    
                    nenotr['range']      = data['data']['range']
                    nenotr['pulsewidth'] = data['data']['pulsewidth']
                    nenotr['baudlength'] = data['data']['txbaud']
                    nenotr['bmcodes']    = data['bmcodes']

                # save data from different DTCs
                for key in array_keys:
                    nenotr[key][i,:,:,j] = output[key]
                for key in sample_keys:
                    nenotr[key][i,:,:,j] = output[key]

                nenotr['txfreq'][i,j] = func(data['txfreq'])
                nenotr['rxfreq'][i,j] = func(data['rxfreq'])

            # assume txpower is the same for each channel
            nenotr['txpower'][i] = func(data['txpower'])
            nenotr['aeurx'][i] = func(data['aeurx'])
            nenotr['aeutx'][i] = func(data['aeutx'])
            nenotr['aeutotal'][i] = func(data['aeutotal'])

        # figure out the average tx and rx frequency
        nenotr['txfreq'] = func(nenotr['txfreq'])
        nenotr['rxfreq'] = func(nenotr['rxfreq'])

        ## NOW CALCULATE DENSITY, SNR, and ERRORS

        # first txpower arrays
        nenotr['txpower'] = np.repeat(nenotr['txpower'][:,np.newaxis],num_beams,axis=1)
        nenotr['txpower'] = np.repeat(nenotr['txpower'][:,:,np.newaxis],num_ranges,axis=2)
        nenotr['txpower'] = np.repeat(nenotr['txpower'][:,:,:,np.newaxis],self.num_freqs,axis=3)

        # ksys
        required_inds = list()
        for bmcode in nenotr['bmcodes'][0,:]:
            required_inds.append(np.where(self.ksys[:,0] == bmcode)[0][0])
        nenotr['ksys'] = self.ksys[required_inds,3]
        nenotr['ksys'] = np.repeat(nenotr['ksys'][:,np.newaxis],num_ranges,axis=1)
        nenotr['ksys'] = np.repeat(nenotr['ksys'][np.newaxis,:,:],num_integrations,axis=0)
        nenotr['ksys'] = np.repeat(nenotr['ksys'][:,:,:,np.newaxis],self.num_freqs,axis=3)
        elevation = self.ksys[required_inds,2] * np.pi / 180.

        # convert received power to nenotr and scale by ambiguity function lag 0
        signal_to_density  = np.ones((num_integrations,num_beams,num_ranges,self.num_freqs))
        ranges = np.repeat(nenotr['range'],num_beams,axis=0)
        ranges = np.repeat(ranges[np.newaxis,:,:],num_integrations,axis=0)
        ranges = np.repeat(ranges[:,:,:,np.newaxis],self.num_freqs,axis=3)
        signal_to_density *= 2.0 * ranges**2 
        signal_to_density /= (nenotr['pulsewidth'] * nenotr['txpower'] * nenotr['ksys'])
        nenotr['density']   = nenotr['signal'] * signal_to_density
        nenotr['density']  /= np.sum(np.abs(data['ambiguity']['wlag'][0,:]))

        # determine the variances for everything
        uncal_signal  = nenotr['uncal_signal']
        noisepower    = nenotr['noisepower']
        noise_samples = nenotr['noisepowersamples']
        power         = nenotr['power']
        power_samples = nenotr['powsamples']
        calplusnoise  = nenotr['calwithnoise']
        cal_samples   = nenotr['calsamples']
        cal           = nenotr['calsignal']

        # variance of noise power estimate
        var_noise  = noisepower**2 / noise_samples
        # variance of power estimates with noise removed
        var_uncal_signal = power**2 / power_samples + var_noise
        var_cal    = calplusnoise**2 / cal_samples + var_noise

        # variance of received (power - noise) / (cal - noise) and density
        var_signal_cal = var_uncal_signal / cal**2 + (uncal_signal**2 * var_cal) / cal**4
        ambiguity_factor = np.sum(np.abs(data['ambiguity']['wlag'][0,:]))
        var_density    = var_signal_cal * signal_to_density**2 / ambiguity_factor**2

        # variance of signal to noise ratio
        var_snr = var_uncal_signal / noisepower**2 + (uncal_signal**2 * var_noise) / noisepower**4

        # save standard deviations for output
        nenotr['edensity'] = np.sqrt(np.sum(var_density,axis=3)/self.num_freqs)
        nenotr['esnr'] = np.sqrt(np.sum(var_snr,axis=3)/self.num_freqs)

        # save average electron density and average snr
        nenotr['density'] = func(nenotr['density'],axis=3)
        nenotr['snr'] = func(nenotr['snr'],axis=3)

        # trim unnecessary dimensions
        nenotr['txpower'] = np.squeeze(nenotr['txpower'][:,0,0,0])

        # convert range into altitude for each beam
        altitude = np.zeros((num_beams,num_ranges))
        elevation = np.repeat(elevation[:,np.newaxis],num_ranges,axis=1)
        for i in range(len(required_inds)):
            altitude[i,:] = nenotr['range'] * np.sin(elevation[i,:])
        nenotr['altitude'] = altitude

        nenotr['bmcodes'] = self.ksys[required_inds,:]

        return nenotr


    @staticmethod
    def get_signal_and_snr(data,config):
        boltzmann = 1.38064852e-23
        # setup the function that will perform mean or median
        if config['nenotr_options']['mean_or_median'] == 'mean':
            func = np.nanmean
        if config['nenotr_options']['mean_or_median'] == 'median':
            func = np.nanmedian

        # cal source power in watts
        cal_source_power = data['ambiguity']['bandwidth'] * data['caltemp'] * boltzmann 

        # get an estimate of the mean/median noise power
        noise      = data['noise']['power']
        noise_pi   = data['noise']['pulsesintegrated']
        num_times, num_beams, num_ranges = noise.shape
        mean_noise = func(func(noise,axis=2)/noise_pi,axis=0)
        num_noise_samples = np.sum(noise_pi, axis=0) * num_ranges

        # get an estimate of the mean/median cal source+noise power (then do measured cal - noise)
        cal      = data['cal']['power']
        cal_pi   = data['cal']['pulsesintegrated']
        num_times, num_beams, num_ranges = cal.shape
        mean_cal_plus_noise = func(func(cal,axis=2)/cal_pi,axis=0)
        num_cal_samples = np.sum(cal_pi,axis=0) * num_ranges
        mean_cal = (mean_cal_plus_noise - mean_noise) / cal_source_power

        # get an estimate of the mean/median signal+noise power (then do sig - noise)
        power      = data['data']['power']
        power_pi   = data['data']['pulsesintegrated']
        num_times, num_beams, num_ranges = power.shape
        power_pi   = np.repeat(power_pi[:,:,np.newaxis],num_ranges,axis=2)
        mean_signal_plus_noise = func(power/power_pi,axis=0)

        # make all arrays have the same dimensions
        mean_noise = np.repeat(mean_noise[:,np.newaxis],num_ranges,axis=1)
        num_noise_samples = np.repeat(num_noise_samples[:,np.newaxis],num_ranges,axis=1)
        mean_cal   = np.repeat(mean_cal[:,np.newaxis],num_ranges,axis=1)
        num_cal_samples = np.repeat(num_cal_samples[:,np.newaxis],num_ranges,axis=1)

        mean_cal_plus_noise = np.repeat(mean_cal_plus_noise[:,np.newaxis],num_ranges,axis=1)

        mean_signal = mean_signal_plus_noise - mean_noise
        num_sig_samples = np.sum(power_pi,axis=0)

        # get snr
        mean_snr = mean_signal / mean_noise

        # get signal and noise in Watts
        mean_signal_cal = mean_signal / mean_cal
        mean_noise_cal  = mean_noise / mean_cal

        output = dict()
        key_list = ['signal','uncal_signal','snr','calsignal','noise','power',
                    'powsamples','calwithnoise','calsamples','noisepower',
                    'noisepowersamples']
        output_list = [mean_signal_cal,mean_signal,mean_snr,mean_cal,
                       mean_noise_cal,mean_signal_plus_noise,num_sig_samples,
                       mean_cal_plus_noise,num_cal_samples,mean_noise,
                       num_noise_samples]
        for key,outarray in zip(key_list,output_list):
            output[key] = outarray

        return output


    def get_site(self):
        site = dict()
        fname = self.datahandlers[0].filelist[0]
        with tables.open_file(fname,'r') as h5:
            site['altitude']  = h5.root.Site.Altitude.read()
            site['code']      = h5.root.Site.Code.read()
            site['latitude']  = h5.root.Site.Latitude.read()
            site['longitude'] = h5.root.Site.Longitude.read()
            site['name']      = h5.root.Site.Name.read()

        return site


    def get_time(self):
        epoch = datetime(1970,1,1)
        time_shape = self.integration_periods.shape
        time = dict()
        keys = ['day','month','year','doy','unixtime']
        for key in keys:
            time[key] = np.zeros(time_shape,dtype=np.int)

        for i,pair in enumerate(self.integration_periods):
            time['day'][i,:]   = np.array([pair[0].day,pair[1].day])
            time['month'][i,:] = np.array([pair[0].month,pair[1].month])
            time['year'][i,:]  = np.array([pair[0].year,pair[1].year])
            time['doy'][i,:]   = np.array([pair[0].timetuple().tm_yday,pair[1].timetuple().tm_yday])
            diff_pair = [(pair[0]-epoch).total_seconds(),(pair[1]-epoch).total_seconds()]
            time['unixtime'][i,:] = diff_pair

        return time


    def run(self):
        # First check if output file is able to be created
        temp_file = tempfile.mktemp()
        output_file = self.config['output']['output_name']

        # Run the calculator and write output to a file
        output = self.calculate_nenotr()

        # get Site information
        site = self.get_site()
        # get Time information
        time = self.get_time()

        # Get current date and time
        date = datetime.utcnow()
        processing_time = date.strftime("%a, %d %b %Y %H:%M:%S +0000")

        # Write the output
        # set up the output file
        print("Writing data to file...")
        with tables.open_file(temp_file,'w') as h5:
            for h5path in h5paths:
                group_path, group_name = os.path.split(h5path[0])
                h5.create_group(group_path,group_name,title=h5path[1],createparents=True)

            node_path = '/ProcessingParams'
            h5.create_array(node_path,'AeuRx',output['aeurx'],createparents=True)
            h5.create_array(node_path,'AeuTotal',output['aeutotal'],createparents=True)
            h5.create_array(node_path,'AeuTx',output['aeutx'],createparents=True)
            h5.create_array(node_path,'BaudLength',output['baudlength'],createparents=True)
            h5.create_array(node_path,'ProcessingTimeStamp',np.array(processing_time),createparents=True)
            h5.create_array(node_path,'PulseLength',output['pulsewidth'],createparents=True)
            h5.create_array(node_path,'RxFrequency',output['rxfreq'],createparents=True)
            h5.create_array(node_path,'TxFrequency',output['txfreq'],createparents=True)
            h5.create_array(node_path,'TxPower',output['txpower'],createparents=True)

            node_path = '/Site'
            h5.create_array(node_path,'Altitude',site['altitude'],createparents=True)
            h5.create_array(node_path,'Code',site['code'],createparents=True)
            h5.create_array(node_path,'Latitude',site['latitude'],createparents=True)
            h5.create_array(node_path,'Longitude',site['longitude'],createparents=True)
            h5.create_array(node_path,'Name',site['name'],createparents=True)

            node_path = '/NeFromPower'
            h5.create_array(node_path,'Altitude',output['altitude'],createparents=True)
            h5.create_array(node_path,'Range',output['range'],createparents=True)
            h5.create_array(node_path,'Ne_NoTr',output['density'],createparents=True)
            h5.create_array(node_path,'errNe_NoTr',output['edensity'],createparents=True)
            h5.create_array(node_path,'SNR',output['snr'],createparents=True)
            h5.create_array(node_path,'errSNR',output['esnr'],createparents=True)

            node_path = '/Time'
            h5.create_array(node_path,'Day',time['day'],createparents=True)
            h5.create_array(node_path,'Month',time['month'],createparents=True)
            h5.create_array(node_path,'Year',time['year'],createparents=True)
            h5.create_array(node_path,'doy',time['doy'],createparents=True)
            h5.create_array(node_path,'UnixTime',time['unixtime'],createparents=True)

            # beamcodes
            h5.create_array('/','BeamCodes',output['bmcodes'],createparents=True)

            # radar mode
            h5.create_array('/','RadarMode',self.mode_name,createparents=True)

        # Add calibration information
        print("Adding calibration information...")
        if self.calibrated:
            cal_file = self.config['input']['ksys_file']
            cal_method = self.config['input']['calibration_method']
        else:
            cal_file = 'None'
            cal_method = 'uncalibrated'
        cal_data = self.ksys
        self.add_calibration_info(temp_file,cal_data,cal_file,cal_method)

        # Add configuration information
        print("Adding configuration information...")
        rawfiles = [x.filelist for x in self.datahandlers]
        self.write_config_info(temp_file,rawfiles)

        with tables.open_file(temp_file,'r+') as h5:
            for key in h5attribs.keys():
                for attr in h5attribs[key]:
                    # print 
                    h5.set_node_attr(key,attr[0],attr[1])
                    # try:  h5.set_node_attr(key,attr[0],attr[1])
                    # except: ''

        # repack the file with compression
        print("Repacking the file with compression...")
        repack = Repackh5(temp_file,output_file)
        repack.repack()
        # remove the temporary file
        print("Cleaning up...")
        os.remove(temp_file)
        print("Making plots...")
        try:
            from .make_nenotr_plots import replot_pcolor_all
            output_dir = self.config['output']['output_path']
            dirname = 'plots_nenotr_' + self.config['default']['integ']
            plots_dir = os.path.join(output_dir,dirname)
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            replot_pcolor_all(output_file,saveplots=1,opath=plots_dir)
        except ImportError as e:
            print("Plotting failed: %s" % str(e))
            pass
        
        print("Done!")



    def add_calibration_info(self,fname,cal_data,cal_fname,cal_method):
        t = datetime.utcnow()
        datestr = t.strftime("%Y-%m-%d")
        # corrChirp=True; chirpCorr=-20.0  #Only used for old RISR-N data that had a transmitter problem

        # Open the fitted h5 file
        with tables.open_file(fname,'r+') as h5:
            # Add the current date to state when the calibration was done
            node_path = '/Calibration'
            h5.create_array(node_path,'CalDate',np.array(datestr),title='Calibration Date',createparents=True)
            # Include the calibration info in the calibration file
            h5.create_array(node_path,'CalDataBeam',cal_data,title='Calibration (ksys) Array',createparents=True)
            # Include the calibration filename
            h5.create_array(node_path,'CalFileBeam',np.array(os.path.basename(cal_fname)),title='Calibration (ksys) filename',createparents=True)
            # Specifiy the calibration method
            h5.create_array(node_path,'CalibrationMethod',np.array(cal_method),title='Method used for calibration.',createparents=True)


    def write_config_info(self,h5name,raw_files):
        import platform
        import getpass

        # Configuration Information
        #Version Number: Follows convention: major.minor.year.month.day
        from . import __version__
        version = __version__

        # Computer information:
        PythonVersion   = platform.python_version()
        Type            = platform.machine()
        System          = "%s %s %s" % (platform.system(),platform.release(),platform.version())
        User            = getpass.getuser()
        Hostname        = platform.node()
        if len(Hostname) == 0:
            import socket
            Hostname = socket.gethostname()

        # Get the config file used
        cf = self.configfile
        Path = os.path.dirname(os.path.abspath(cf))
        Name = os.path.basename(cf)

        with open(cf,'r') as f:
            Contents = "".join(f.readlines())

        # Record the raw files used
        # Make a string listing all the files
        RawFiles = ''
        for i,files in enumerate(raw_files):
            temp = "\n".join(files)
            if i != 0:
                RawFiles += '\n'
            RawFiles += temp

        # Record the directory where fitted files can be found
        OutputPath = os.path.abspath(self.config['output']['output_path'])

        # Open the fitted h5 file
        with tables.open_file(h5name,'r+') as h5:
            node_path = '/ProcessingParams'
            h5.create_group(node_path,'ComputerInfo',title='Processing Computer Information',createparents=True)
            h5.create_group(node_path,'ConfigFiles',title='Config File Information',createparents=True)
            h5.create_array(node_path,'SoftwareVersion',np.array(version),title='Version of software that made this file',createparents=True)
            h5.create_array(node_path,'RawFiles',np.array(RawFiles),title='The raw files used to produce this file',createparents=True)
            h5.create_array(node_path,'OutputPath',np.array(OutputPath),title='Path where this file was originally made',createparents=True)
            node_path = '/ProcessingParams/ComputerInfo'
            h5.create_array(node_path,'PythonVersion',np.array(PythonVersion),title='Version of python used',createparents=True)
            h5.create_array(node_path,'Type',np.array(Type),title='Type of operating system',createparents=True)
            h5.create_array(node_path,'System',np.array(System),title='System information',createparents=True)
            h5.create_array(node_path,'User',np.array(User),title='Username',createparents=True)
            h5.create_array(node_path,'Host',np.array(Hostname),title='Hostname of the computer',createparents=True)
            node_path = '/ProcessingParams/ConfigFiles/File1'
            h5.create_array(node_path,'Name',np.array(Name),createparents=True)
            h5.create_array(node_path,'Path',np.array(Path),createparents=True)
            h5.create_array(node_path,'Contents',np.array(Contents),createparents=True)


config_file_help = """Calculate electron density from level 1 coherently coded data
files. The code requires power estimates of signal, noise, and cal.

Requires a configuration file containing the following example format:
[DEFAULT]
#optional variables to use in substitutions below
EXPNAME=20171003.001
INTEG=20sec
[NEPOW_OPTIONS]
#number of seconds of data to integrate
Recs2integrate=20
#use mean or median
mean_or_median=median
[INPUT]
#input paths (separate frequencies with comma,
# separate searches within same frequency with colons)
# example 2 frequency, 2 search path per frequency
file_paths=/path1/*.dt0.h5:/path2/*.dt0.h5,/path1/*.dt1.h5:/path2/*.dt1.h5
# Optional: Path to file containing lag ambiguity function
AMB_PATH=/home/asreimer/temp/ne_pow/AmbFunc.h5
# Optional: Path to the system constant file. If not provided, loaded from data
#ksys_file=/home/asreimer/temp/ne_pow/cal-201710-calibration-ksys-10.05.2017.txt
#calibration_method=plasma line
[OUTPUT]
# Output path
OUTPUT_PATH=/path/to/output/directory/%%(EXPNAME)s
# Output filename
OUTPUT_NAME=%%(OUTPUT_PATH)s/%%(EXPNAME)s_nenotr_%%(INTEG)s.h5
"""


# a function to run this file as a script
def main():
    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    # Build the argument parser tree
    parser = ArgumentParser(description=config_file_help,
                            formatter_class=RawDescriptionHelpFormatter)
    arg = parser.add_argument('config_file',help='A configuration file.')
    
    args = vars(parser.parse_args())
    nenotr = CalcNeNoTr(args['config_file'])
    nenotr.run()



# Run as a script
if __name__ == '__main__':
    main()
