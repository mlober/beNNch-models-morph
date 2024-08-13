import os
import sys
import time
import nest

import numpy as np


M_INFO = 10
M_ERROR = 30

class Model:

    def __init__(self, params):

        print('\nInitialising model and simulation...')

        self.params = params

        
        self.network_params = params['network_params']
        self.model_params = params['model_params']
        if self.model_params['phase'] == 'uniform':
            self.model_params['phase'] = nest.random.uniform(0, 1)

        self.num_areas = params['num_areas']

    def __setup_nest(self):

        data_dir = self.params['data_path']
        self.data_path = os.path.join(data_dir, 'recordings')

        nest.ResetKernel()
        nest.SetKernelStatus({
            'total_num_virtual_procs': self.params['nvp'],
            'resolution': self.params['dt'],
            'min_delay': self.params['min_delay'],
            'max_delay': self.params['max_delay'],
            'rng_seed': self.params['rng_seed'],
            'spike_buffer_shrink_limit': 0.2#,
            #'use_compressed_spikes': False
            })
        nest.set_verbosity(M_INFO)

        if self.params['morph']:
            nest.SetKernelStatus({'threshold_delay': self.params['threshold_delay']})

    def create(self):
        
        tic = time.time()  # start timer on construction 
        self.neurons = {}

        if self.params['morph'] == True:
            self.__create_neurons()
            for idx, area in enumerate(self.params['areas_list']):
                self.neurons[area] = self.__define_area(area, idx)
        else:
            for area in self.params['areas_list']:
                self.neurons[area] = self.__create_area(area)

        if self.params['record_spikes'] == True:
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

        self.recorder = nest.Create('spike_recorder', params={'record_to': 'ascii', 'label': recorder_label})

    def connect(self):

        tic = time.time()

        nest.message(M_INFO, 'build_network', 'Connecting stimulus generators.')
        
        # Connections within area

        nest.message(M_INFO, 'build_network',
                    'Connecting the population internally.')
        
        #delay_mean_intra = (self.params['threshold_delay'] + self.params['min_delay']) / 2
        delay_mean_intra = self.params['delay_mean_intra']
        for area in self.params['areas_list']:
            nest.Connect(self.neurons[area], self.neurons[area],
                        {'rule': 'fixed_indegree', 'indegree': self.network_params[area]['indegree']},
                        {'synapse_model': 'static_synapse', 'weight': 0., 
                         'delay': nest.math.redraw(nest.random.normal(delay_mean_intra, 0.5*delay_mean_intra), min=self.params['dt'], max=self.params['max_delay'])})
        
        #delay_mean_inter =  (self.params['max_delay'] + self.params['threshold_delay']) / 2 
        delay_mean_inter = self.params['delay_mean_inter']
        for area_pre in self.params['areas_list']:
            for area_post in self.params['areas_list']:
                if area_pre != area_post:
                    if self.params['morph']:
                        conn_dict = {'rule': 'fixed_indegree', 'indegree': self.network_params[area_pre][area_post]['indegree'],
                                     'long_range': self.params['morph']}
                    else:
                        conn_dict = {'rule': 'fixed_indegree', 'indegree': self.network_params[area_pre][area_post]['indegree']}
                    nest.Connect(self.neurons[area_pre], self.neurons[area_post], conn_dict,
                                 {'synapse_model': 'static_synapse', 'weight': 0., 
                                  'delay': nest.math.redraw(nest.random.normal(delay_mean_inter, 0.5*delay_mean_inter), min=self.params['threshold_delay'], max=self.params['max_delay'])}
                                )

        self.BuildEdgeTime = time.time() - tic


    def logging_presim(self):
        timer_keys = ['time_collocate_spike_data',
                      'time_communicate_spike_data',
                      'time_communicate_spike_data_local',
                      'time_communicate_spike_data_global',
                      'time_deliver_spike_data',
                      'time_gather_spike_data',
                      'time_update',
                      'time_simulate',
                      'time_synch_global'
                      ]
        values = nest.GetKernelStatus(timer_keys)

        self.presim_timers = dict(zip(timer_keys, values))

        fn = os.path.join(self.data_path,
                              '_'.join(('logfile',
                                        str(nest.Rank()))))
        with open(fn, 'w') as f:
            for idx, value in enumerate(values):
                f.write('presim_' + timer_keys[idx] + ' ' + str(value) + '\n')
            f.write('presim_local_spike_counter' + ' ' + str(nest.GetKernelStatus('local_spike_counter')) + '\n')

    def logging(self):
        d = {'py_time_kernel_prepare': self.time_kernel_prepare,
             'py_time_presimulate': self.time_presimulate,
             'py_time_network_prepare': self.time_network_prepare,
             'py_time_simulate': self.time_simulate,
             'py_time_create': self.time_create,
             'py_time_connect': self.time_connect,
             'base_memory': self.base_memory,
             'network_memory': self.network_memory,
             'init_memory': self.init_memory,
             'total_memory': self.total_memory}
        d.update(nest.GetKernelStatus())

        # subtract presim timers from simtime timers
        for key in self.presim_timers.keys():
            d[key] -= self.presim_timers[key]

        fn = os.path.join(self.data_path,
                              '_'.join(('logfile',
                                        str(nest.Rank()))))

        with open(fn, 'a') as f:
            for key, value in d.items():
                f.write(key + ' ' + str(value) + '\n')

        fn_cycle_time = os.path.join(self.data_path,
                                     '_'.join(('cycle_time_log',
                                               str(nest.Rank()))))

        np.savetxt(fn_cycle_time, np.transpose([d['cycle_time_log']['times'], d['cycle_time_log']['communicate_time'],
                                                d['cycle_time_log']['communicate_time_global'], d['cycle_time_log']['communicate_time_local'],
                                                d['cycle_time_log']['synch_time'], d['cycle_time_log']['local_spike_counter']]))

    def memory(self):
        """
        Use NEST's memory wrapper function to record used memory.
        """
        mem = nest.ll_api.sli_func('memory_thisjob')
        if isinstance(mem, dict):
            return mem['heap']
        else:
            return mem

    def simulate(self):
        t0 = time.time()
        self.base_memory = self.memory()
        self.__setup_nest()
        t1 = time.time()
        self.time_kernel_prepare = t1 - t0

        self.create()
        self.time_create = time.time() - t1
        
        t2 = time.time()
        self.connect()
        self.network_memory = self.memory()
        self.time_connect = time.time() - t2

        t3 = time.time()
        nest.Prepare()
        self.time_network_prepare = time.time() - t3

        t4 = time.time()
        nest.Run(self.params['presimtime'])
        self.init_memory = self.memory()
        self.time_presimulate = time.time() - t4
        self.logging_presim()

        if self.params['record_spikes']:
            area = self.params['areas_list'][0]
            if len(self.neurons[area]) < self.network_params[area]['Nrec']:
                nest.message(
                    M_ERROR, 'build_network',
                    """Spikes can only be recorded from local neurons, but the
                    number of local neurons is smaller than the number of neurons
                    spikes should be recorded from. Aborting the simulation!""")
                exit(1)
            nc_sliceable = nest.NodeCollection(self.neurons[area].tolist())
            nest.message(M_INFO, 'build_network', 'Connecting spike recorders.')
            nest.Connect(nc_sliceable[:self.network_params[area]['Nrec']], self.recorder)

        t5 = time.time()
        nest.Run(self.params['simtime'])
        self.time_simulate = time.time() - t5

        self.total_memory = self.memory()
        self.logging()

