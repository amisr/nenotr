# -*- coding: utf-8 -*-
# Created: 4 July 2018
# Author: Ashton S. Reimer

import re
import os
import tables
import numpy as np
from datetime import datetime


nenotr_modes = ['/CohCode']
class DataHandler(object):
    def __init__(self,filelist):

        # Input checking
        if not len(filelist):
            raise Exception('File list is empty.')
        self.filelist = filelist
        
        # check for PLFFTS in the files
        self.nenotr_mode = self.__which_nenotr_mode(self.filelist[0])
        if self.nenotr_mode is None:
            raise Exception('No supported dregion mode found from: %s \nAre these dregion mode files?' % str(nenotr_modes))

        # detect which DTC the files are from (expect all files are from same dtc)
        self.dtc = self.__which_dtc()

        # read times from the files
        self.__catalog_data()

        # determine the array shapes in the files
        self.__determine_array_sizes()

        # initialize file handling variables
        self.__loaded_file = None
        self.__loaded_data = None
        self.__loaded_beamcodes = None

    def get_record(self,requested_time):
        if self.__load_file(requested_time) is None:
            return None

        time_ind = np.where((self.__loaded_time[:,0] <= requested_time) & (self.__loaded_time[:,1] > requested_time))[0]
        
        # it's possible that we get a requested time where there's no valid data (14 time records, but only 13 data records...)
        # so we need to check if time_ind is empty and handle this case
        if len(time_ind) == 0:
            return None

        data = dict()
        data['data'] = dict()
        data['noise'] = dict()
        data['cal'] = dict()
        data['data']['range']       = self.__loaded_data['data']['range']
        data['data']['pulsewidth']  = self.__loaded_data['data']['pulsewidth']
        data['data']['txbaud']      = self.__loaded_data['data']['txbaud']
        data['noise']['pulsewidth'] = self.__loaded_data['noise']['pulsewidth']
        data['noise']['txbaud']     = self.__loaded_data['noise']['txbaud']
        data['cal']['pulsewidth']   = self.__loaded_data['cal']['pulsewidth']
        data['cal']['txbaud']       = self.__loaded_data['cal']['txbaud']

        data['data']['power']             = np.squeeze(self.__loaded_data['data']['power'][time_ind,:])
        data['data']['pulsesintegrated']  = np.squeeze(self.__loaded_data['data']['pulsesintegrated'][time_ind,:])
        data['noise']['power']            = np.squeeze(self.__loaded_data['noise']['power'][time_ind,:])
        data['noise']['pulsesintegrated'] = np.squeeze(self.__loaded_data['noise']['pulsesintegrated'][time_ind,:])
        data['cal']['power']              = np.squeeze(self.__loaded_data['cal']['power'][time_ind,:])
        data['cal']['pulsesintegrated']   = np.squeeze(self.__loaded_data['cal']['pulsesintegrated'][time_ind,:])

        data['txfreq']    = np.squeeze(self.__loaded_data['txfreq'][time_ind,:])
        data['txpower']   = np.squeeze(self.__loaded_data['txpower'][time_ind,:])
        data['rxfreq']    = np.squeeze(self.__loaded_data['rxfreq'][time_ind,:])
        data['aeurx']     = np.squeeze(self.__loaded_data['aeurx'][time_ind,:])
        data['aeutx']     = np.squeeze(self.__loaded_data['aeutx'][time_ind,:])
        data['aeutotal']  = np.squeeze(self.__loaded_data['aeutotal'][time_ind,:])

        data['caltemp'] = self.__loaded_data['caltemp']
        data['bmcodes'] = self.__loaded_beamcodes

        return data

    def get_records(self,start_time,end_time):

        # figure out how many time records we have to get
        # the logic on the next line is correct, even though it seems confusing at first
        # The start time needs to be checked against the end times of each time record
        # and the e time needs to be checked against the start times of each record.
        request_time_inds = np.where((self.times[:,0] <= end_time) & (self.times[:,1] >= start_time))[0]
        temp_times = self.times[request_time_inds]

        if len(temp_times) < 1:
            print("No data for request start and end times.")
            return None

        request_times = list()
        epoch = datetime(1970,1,1)
        for tup in temp_times:
            t1 = (tup[0] - epoch).total_seconds()
            t2 = (tup[1] - epoch).total_seconds()
            request_times.append(datetime.utcfromtimestamp((t1 + t2) / 2.0))
        request_times.sort()

        num_times = len(request_times)

        arrsh = self.__array_shapes
        data = dict()
        data['data'] = dict()
        data['noise'] = dict()
        data['cal'] = dict()
        data['data']['power']             = np.nan*np.zeros((num_times,) + arrsh['data']['power'][1:])
        data['data']['range']             = np.nan*np.zeros(arrsh['data']['range'])
        data['data']['pulsesintegrated']  = np.nan*np.zeros((num_times,) + arrsh['data']['pulsesintegrated'][1:])

        data['noise']['power']            = np.nan*np.zeros((num_times,) + arrsh['noise']['power'][1:])
        data['noise']['pulsesintegrated'] = np.nan*np.zeros((num_times,) + arrsh['noise']['pulsesintegrated'][1:])

        data['cal']['power']              = np.nan*np.zeros((num_times,) + arrsh['cal']['power'][1:])
        data['cal']['pulsesintegrated']   = np.nan*np.zeros((num_times,) + arrsh['cal']['pulsesintegrated'][1:])
        data['txfreq']                    = np.nan*np.zeros((num_times,2))
        data['txpower']                   = np.nan*np.zeros((num_times,2))
        data['rxfreq']                    = np.nan*np.zeros((num_times,2))
        data['aeurx']                     = np.nan*np.zeros((num_times,2))
        data['aeutx']                     = np.nan*np.zeros((num_times,2))
        data['aeutotal']                  = np.nan*np.zeros((num_times,2))

        # now get the data for the requested time
        for i,time in enumerate(request_times):
            temp = self.get_record(time)
            if temp is None:
                continue
            
            data['data']['range']       = temp['data']['range']
            data['data']['pulsewidth']  = temp['data']['pulsewidth']
            data['data']['txbaud']      = temp['data']['txbaud']
            data['noise']['pulsewidth'] = temp['noise']['pulsewidth']
            data['noise']['txbaud']     = temp['noise']['txbaud']
            data['cal']['pulsewidth']   = temp['cal']['pulsewidth']
            data['cal']['txbaud']       = temp['cal']['txbaud']
            data['caltemp']             = temp['caltemp']
            data['bmcodes']             = temp['bmcodes']

            data['data']['power'][i,:]             = temp['data']['power']
            data['data']['pulsesintegrated'][i,:]  = temp['data']['pulsesintegrated']
            data['noise']['power'][i,:]            = temp['noise']['power']
            data['noise']['pulsesintegrated'][i,:] = temp['noise']['pulsesintegrated']
            data['cal']['power'][i,:]              = temp['cal']['power']
            data['cal']['pulsesintegrated'][i,:]   = temp['cal']['pulsesintegrated']
            data['txfreq'][i,:]                    = temp['txfreq']
            data['txpower'][i,:]                   = temp['txpower']
            data['rxfreq'][i,:]                    = temp['rxfreq']
            data['aeurx'][i,:]                     = temp['aeurx']
            data['aeutx'][i,:]                     = temp['aeutx']
            data['aeutotal'][i,:]                  = temp['aeutotal']

        return data, temp_times


    def __load_file(self,requested_time):

        time_ind = np.where((self.times[:,0] <= requested_time) & (self.times[:,1] >= requested_time))[0]
        if len(time_ind) < 1:
            print("Requested time not found.")
            return None

        # If it is, let's see if we have that file loaded or not
        needed_file = self.filetimes[tuple(self.times[time_ind[0]])]
        if self.__loaded_file != needed_file:
            print("Loading file: %s" % needed_file)
            # Load the arrays in Data/Spectra and the Time/UnixTime data
            temp_data = dict()
            temp_data['data'] = dict()
            temp_data['noise'] = dict()
            temp_data['cal'] = dict()
            with tables.open_file(needed_file,'r') as h5:
                # Handle beamcodes for AMISR and Sondrestrom
                try:
                    bmcode_path = "%s/Data/" % self.nenotr_mode
                    node = h5.get_node(bmcode_path)
                    self.__loaded_beamcodes = np.array(node.Beamcodes.read())
                except tables.NoSuchNodeError:
                    # SONDRESTROM NO SUPPORTED HERE. NEED TO ADD.
                    raise NotImplemented("Sondrestrom data not supported.")
                    # az = np.mean(output1['/Antenna']['Azimuth'])
                    # el = np.mean(output1['/Antenna']['Elevation'])
                    # self.__loaded_beamcodemap = np.array([[32768,az,el,0.0]])

                node = h5.get_node(self.nenotr_mode)
                temp_data['data']['power'] = node.Data.Power.Data.read()
                temp_data['data']['range'] = node.Data.Power.Range.read()
                temp_data['data']['pulsesintegrated'] = node.Data.PulsesIntegrated.read()
                temp_data['data']['pulsewidth'] = node.Data.Pulsewidth.read() 
                temp_data['data']['txbaud'] = node.Data.TxBaud.read()

                temp_data['noise']['power'] = node.Noise.Power.Data.read()
                temp_data['noise']['pulsesintegrated'] = node.Noise.PulsesIntegrated.read()
                temp_data['noise']['pulsewidth'] = node.Noise.Pulsewidth.read() 
                temp_data['noise']['txbaud'] = node.Noise.TxBaud.read()

                temp_data['cal']['power'] = node.Cal.Power.Data.read()
                temp_data['cal']['pulsesintegrated'] = node.Cal.PulsesIntegrated.read()
                temp_data['cal']['pulsewidth'] = node.Cal.Pulsewidth.read() 
                temp_data['cal']['txbaud'] = node.Cal.TxBaud.read()

                temp_data['caltemp']  = h5.root.Rx.CalTemp.read()
                temp_data['txfreq']   = h5.root.Tx.Frequency.read()
                temp_data['txpower']  = h5.root.Tx.Power.read()
                temp_data['rxfreq']   = h5.root.Rx.Frequency.read()
                # handle old files with aeurx and tx missing
                try:
                    temp_data['aeurx']    = h5.root.Rx.AeuRx.read()
                except tables.NoSuchNodeError:
                    temp_data['aeurx']    = temp_data['txpower'] * 0.0
                try:
                    temp_data['aeutx']    = h5.root.Rx.AeuTx.read()
                except tables.NoSuchNodeError:
                    temp_data['aeutx']    = temp_data['txpower'] * 0.0
                try:
                    temp_data['aeutotal']    = h5.root.Rx.AeuTotal.read()
                except tables.NoSuchNodeError:
                    temp_data['aeutotal']    = temp_data['txpower'] * 0.0

                # Time
                temp_time = h5.root.Time.UnixTime.read()

            self.__loaded_data = temp_data

            # Check to make sure data, noise, cal power, and
            # unix_time all have the same number of time records
            pow_times     = temp_data['data']['power'].shape[0]
            noise_times   = temp_data['noise']['power'].shape[0]
            cal_times     = temp_data['cal']['power'].shape[0]
            txfreq_times  = temp_data['txfreq'].shape[0]
            rxfreq_times  = temp_data['rxfreq'].shape[0]
            txpower_times = temp_data['txpower'].shape[0]
            aeutx_times   = temp_data['aeutx'].shape[0]
            aeurx_times   = temp_data['aeurx'].shape[0]
            aeutot_times  = temp_data['aeutotal'].shape[0]
            time_times    = temp_time.shape[0]
            num_times     = np.min([pow_times,noise_times,cal_times,time_times,
                                    txfreq_times,rxfreq_times,txpower_times,
                                    aeutx_times,aeurx_times,aeutot_times])

            # ensure all arrays only have the proper time array shape
            temp_data['data']['power']  = temp_data['data']['power'][:num_times,:]
            temp_data['noise']['power'] = temp_data['noise']['power'][:num_times,:]
            temp_data['cal']['power']   = temp_data['cal']['power'][:num_times,:]
            temp_data['data']['pulsesintegrated']  = temp_data['data']['pulsesintegrated'][:num_times,:]
            temp_data['noise']['pulsesintegrated'] = temp_data['noise']['pulsesintegrated'][:num_times,:]
            temp_data['cal']['pulsesintegrated']   = temp_data['cal']['pulsesintegrated'][:num_times,:]
            temp_data['txfreq']   = temp_data['txfreq'][:num_times,:]
            temp_data['txpower']  = temp_data['txpower'][:num_times,:]
            temp_data['rxfreq']   = temp_data['rxfreq'][:num_times,:]
            temp_data['aeurx']    = temp_data['aeurx'][:num_times,:]
            temp_data['aeutx']    = temp_data['aeutx'][:num_times,:]
            temp_data['aeutotal'] = temp_data['aeutotal'][:num_times,:]
            temp_time = temp_time[:num_times,:]

            # Convert UnixTime to datetime
            temp = list()
            for i in range(num_times):
                time_pair = [datetime.utcfromtimestamp(temp_time[i,0]),
                             datetime.utcfromtimestamp(temp_time[i,1])]
                temp.append(time_pair)
            self.__loaded_time = np.array(temp)
            self.__loaded_file = needed_file

        return True


    def __determine_array_sizes(self):
        temp_shape = dict()
        temp_shape['data'] = dict()
        temp_shape['noise'] = dict()
        temp_shape['cal'] = dict()
        with tables.open_file(self.filelist[0],'r') as h5:
            node = h5.get_node(self.nenotr_mode)
            temp_shape['data']['power'] = node.Data.Power.Data.shape
            temp_shape['data']['range'] = node.Data.Power.Range.shape
            temp_shape['data']['pulsesintegrated'] = node.Data.PulsesIntegrated.shape
            temp_shape['data']['pulsewidth'] = node.Data.Pulsewidth.shape 
            temp_shape['data']['txbaud'] = node.Data.TxBaud.shape

            temp_shape['noise']['power'] = node.Noise.Power.Data.shape
            temp_shape['noise']['pulsesintegrated'] = node.Noise.PulsesIntegrated.shape
            temp_shape['noise']['pulsewidth'] = node.Noise.Pulsewidth.shape 
            temp_shape['noise']['txbaud'] = node.Noise.TxBaud.shape

            temp_shape['cal']['power'] = node.Cal.Power.Data.shape
            temp_shape['cal']['pulsesintegrated'] = node.Cal.PulsesIntegrated.shape
            temp_shape['cal']['pulsewidth'] = node.Cal.Pulsewidth.shape 
            temp_shape['cal']['txbaud'] = node.Cal.TxBaud.shape

        self.__array_shapes = temp_shape


    def __catalog_data(self):
        # for each file in self.filelist, grab all the times and make a dictionary mapping to that file for each datetime

        self.filetimes = dict()

        # For every file that we have, grab the start and end times in the files.
        # We need to check every time, in case there is a gap in the file. Also
        # should make sure power, noise, cal, and unix time have same # of time records
        for fname in self.filelist:
            with tables.open_file(fname,'r') as h5:
                node = h5.get_node(self.nenotr_mode)
                temp_times = h5.root.Time.UnixTime.read()
                pow_times   = node.Data.Power.Data.shape[0]
                noise_times = node.Noise.Power.Data.shape[0]
                cal_times   = node.Cal.Power.Data.shape[0]
                time_times  = temp_times.shape[0]

            num_times   = np.min([pow_times,noise_times,cal_times,time_times])

            for i in range(num_times):
                file_time = (datetime.utcfromtimestamp(temp_times[i,0]),
                             datetime.utcfromtimestamp(temp_times[i,1])
                            )
                self.filetimes[file_time] = fname

        # now get an array of start and ends times from the keys of the filetimes dict
        temp = [list(x) for x in self.filetimes.keys()]
        temp.sort()

        self.times = np.array(temp)


    # Figure out which plasma line mode was run
    def __which_nenotr_mode(self,h5file):
        # Check a file to see which plasma line mode it was running
        with tables.open_file(h5file) as h5:
            for nenotr_mode in nenotr_modes:
                try:
                    # If we found the correct mode name, return it
                    h5.get_node(nenotr_mode)
                    return nenotr_mode
                except tables.NoSuchNodeError:
                    pass
        # If no valid mode is found, then we return None.
        return None


    # Figure out which DTC this was run on. Need this for looking up information in the exp file.
    def __which_dtc(self):
        # Get this info from the file name. Terrible, but there is no easy and uniquely indentifying info in the raw files
        # one strategy would be to get info from the exp file and try to match it up with info in the h5 file. Hard, not necessarily unique.

        dtc = None

        filename = os.path.basename(self.filelist[0])
        m = re.search('dt(.+?).h5',filename)

        if m:
            dtc = int(m.group(1))

        return dtc
