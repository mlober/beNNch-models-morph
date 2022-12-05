import os
import sys
import time
import nest
import helper

import numpy as np


M_INFO = 10
M_ERROR = 30

class Model:

    def __init__(self, params):

        print('\nInitialising model and simulation...')

        self.params = params

        
        self.network_params = params['network_params']
        self.model_params = params['model_params']

        self.num_areas = params['num_areas']

        self.__setup_nest()

    def __setup_nest(self):

        data_dir = os.path.join(self.params['data_path'], self.params['data_folder_hash'])

        nest.ResetKernel()
        nest.SetKernelStatus({
            'total_num_virtual_procs': self.params['nvp'],
            'resolution': self.params['dt'],
            'min_delay': self.params['min_delay'],
            'max_delay': self.params['max_delay'],
            'rng_seed': self.params['rng_seed'],
            'overwrite_files': True, 
            'data_path': os.path.join(data_dir, 'recordings')})
        nest.set_verbosity(M_INFO)

        if self.params['morph_distribution']:
            nest.SetKernelStatus({'threshold_delay': self.params['threshold_delay']})

    def create(self):
        
        tic = time.time()  # start timer on construction 
        self.neurons = {}

        if self.params['morph_distribution']:
            self.__create_neurons()
            for area_idx, area in enumerate(self.params['areas_list']):
                self.neurons[area] = self.__define_area(area, area_idx)
        else:
            for area in self.params['areas_list']:
                self.neurons[area] = self.__create_area(area)

        self.__create_spike_recorder()

        self.BuildNodeTime = time.time() - tic

    def __create_area(self, area):
        neurons = nest.Create('ignore_and_fire', self.network_params[area]['N_total'], params=self.model_params)
        return neurons

    def __create_neurons(self):
        total_num_neurons_per_area = [self.network_params[a]['N_total'] for a in self.params['areas_list']]
        max_num_neurons_per_area = int(max(total_num_neurons_per_area))

        total_num_neurons = max_num_neurons_per_area * self.num_areas
        self.all_neurons = nest.Create('ignore_and_fire', total_num_neurons, params=self.model_params)
        nest.SetStatus(self.all_neurons, {'frozen': True})
        
    def __define_area(self, area, area_idx):

        nest.message(M_INFO, 'build_network', 'Creating excitatory population.')

        neurons = self.all_neurons[area_idx:self.network_params[area]['N_total']*self.num_areas:self.num_areas]
        nest.SetStatus(neurons, 'frozen', False)

        return neurons

    def __create_spike_recorder(self):

        nest.message(M_INFO, 'build_network',
                 'Creating excitatory spike recorder.')

        recorder_label = os.path.join(
            self.params['path_name'],
            'alpha_' + '_spikes')

        self.recorder = nest.Create('spike_recorder') #, params={'record_to': 'ascii', 'label': recorder_label})

    def __define_delay_distr(self):

        self.delay_dist_within = {}
        for area in self.params['areas_list']:
            d_within = nest.math.redraw(nest.random.normal(mean=self.network_params[area]['delay_mean'], 
            std=self.network_params[area]['delay_rel_std']*self.network_params[area]['delay_mean']), min=self.params['min_delay'], max=self.params['threshold_delay'])
            self.delay_dist_within[area] = d_within
 
        self.delay_dist_between = {}
        for area_pre in self.params['areas_list']:
            self.delay_dist_between[area_pre] = {}
            for area_post in self.params['areas_list']:
                if area_pre != area_post:      
                    d_between = nest.math.redraw(nest.random.normal(mean=self.network_params[area_pre][area_post]['delay_mean'], 
                    std=self.network_params[area_pre][area_post]['delay_rel_std']*self.network_params[area_pre][area_post]['delay_mean']), min=self.params['threshold_delay'], max=self.params['max_delay'])
                    self.delay_dist_between[area_pre][area_post] = d_between

    def connect(self):

        self.__define_delay_distr()

        tic = time.time()

        nest.message(M_INFO, 'build_network', 'Connecting stimulus generators.')
        
        # Connections within area

        nest.message(M_INFO, 'build_network',
                    'Connecting the population internally.')
                    
        for area in self.params['areas_list']:
            nest.Connect(self.neurons[area], self.neurons[area],
                        {'rule': 'fixed_indegree', 'indegree': self.network_params[area]['indegree']},
                        {'synapse_model': 'static_synapse', 'weight': 0., 
                        'delay': self.delay_dist_within[area]})
            print('internally connecting ', area)

        #conn = nest.GetConnections(self.E_neurons['area_1'])
        #print(conn.get('delay')[:10])
          
        for area_pre in self.params['areas_list']:
            for area_post in self.params['areas_list']:
                if area_pre != area_post:
                    nest.Connect(self.neurons[area_pre], self.neurons[area_post],
                                {'rule': 'fixed_indegree', 'indegree': self.network_params[area_pre][area_post]['indegree']},
                                {'synapse_model': 'static_synapse', 'weight': 0., 
                                'delay': self.delay_dist_between[area_pre][area_post]})
                    print(f'connecting {area_pre} to {area_post}')


        self.BuildEdgeTime = time.time() - tic


    def simulate(self):

        tic_presim = time.time()

        nest.Simulate(self.params['presimtime'])

        self.PreSimCPUTime = time.time() - tic_presim
        
        if self.params['record_spikes']:
            area = self.params['areas_list'][0]
            if self.params['nvp'] != 1:
                nc_sliceable = nest.NodeCollection(self.neurons[area].tolist())
                local_neurons = nest.GetLocalNodeCollection(nc_sliceable)
                # GetLocalNodeCollection returns a stepped composite NodeCollection, which
                # cannot be sliced. In order to allow slicing it later on, we're creating a
                # new regular NodeCollection from the plain node IDs.
                local_neurons = nest.NodeCollection(local_neurons.tolist())
            else:
                local_neurons = nest.NodeCollection(self.neurons[area].tolist())

            if len(local_neurons) < self.network_params[area]['Nrec']:
                nest.message(
                    M_ERROR, 'build_network',
                    """Spikes can only be recorded from local neurons, but the
                    number of local neurons is smaller than the number of neurons
                    spikes should be recorded from. Aborting the simulation!""")
                exit(1)

            nest.message(M_INFO, 'build_network', 'Connecting spike recorders.')
            nest.Connect(local_neurons[:self.network_params[area]['Nrec']], self.recorder)


        tic = time.time()

        nest.Simulate(self.params['simtime'])

        self.SimCPUTime = time.time() - tic

        average_rate = 0.0
        sr = self.recorder if self.params['record_spikes'] else None

        if self.params['record_spikes']:
            average_rate = helper.compute_rate(self.params['original']['withinarea']['Nrec'], self.params['simtime'], sr)
            #fano_factor = helper.compute_fano_factor(sr)

        # helper.plot_spikes_hist(sr, self.params['simtime'])

        d = {'network_size': nest.GetKernelStatus('network_size'),
             'num_connections': nest.GetKernelStatus('num_connections'),
             'py_time_create': self.BuildNodeTime,
             'py_time_connect': self.BuildEdgeTime,
             'py_time_presimulate': self.PreSimCPUTime,
             'py_time_simulate': self.SimCPUTime,
             'average_rate': average_rate}
        print(d)

