#import parameters as para
import numpy as np
import copy
import nest


p={}

p['data_path'] = './data'

p['morph'] = True
p['threshold_delay'] = 1.0

p['num_processes'] = 32
p['num_areas'] = 32
p['threads_per_task'] = 64
# p['nvp'] = num_mpi * num_local_threads

p['scale'] = 1.0                        # scaling factor of the network size

p['presimtime'] = 1000                   # total network size = scale*11250 neurons
p['simtime'] = 10000                     # total simulation time in ms
p['dt'] = 0.1                           # simulation step
                        # switch to record spikes of excitatory
p['record_spikes'] = False
                                        # neurons to file
p['rng_seed'] = 12                      # random number generator seed
p['numpy_seed'] = 20                    # random number generator seed for numpy, used for drawing area sizes
p['path_name'] ='.'                     # path where all files will have to be written
p['log_file'] = 'logfile'               # naming scheme for the log files

p['min_delay'] = 0.1
p['max_delay'] = 2.0
p['delay_mean_intra'] = 0.5
p['delay_mean_inter'] = 1.5

p['area_size'] = 129068 # mean area size of macaque multi-area model

p['unbalanced_network_sigma'] = 0.15

p['unbalanced_activity_sigma'] = 0.
p['unbalanced_activity_mu'] = 14.7 # from MAM metastable

############# Model parameters ##############################################

p['model_params'] = {}
p['model_params']['rate'] = 2.5
p['model_params']['phase'] = 'uniform'

############# Network parameters ############################################

def calc_dependend_parameters(p):
    rng = np.random.default_rng(p['numpy_seed'])
    p['areas_list'] = []
    for i in range(p['num_areas']):
        string = 'area_' + str(i)
        p['areas_list'].append(string)
    
    p['original'] = {}
    p['original']['withinarea'] = {}
    p['original']['withinarea']['N_total'] = int(p['area_size'] * p['scale'])
    p['original']['withinarea']['Nrec'] = int(min(p['original']['withinarea']['N_total'], 10000))         # number of neurons to record spikes from
    #p['original']['withinarea']['indegree'] = int(0.05 * p['original']['withinarea']['N_total'])         # total num connections per neuron
    p['original']['withinarea']['indegree'] = 2921 # half of mean indegree of macaque multi-area model (half of incoming connections from within area, other half from other areas)
  
    p['original']['interareal'] = {}
    if p['num_areas'] > 1:
        p['original']['interareal']['indegree'] = int(p['original']['withinarea']['indegree'] * 1/(p['num_areas'] - 1))
    
    
    ############# Set all parameters to default "original"-Parameters ###########
    
    p['network_params'] = {}
    for area_pre in p['areas_list']:
        p['network_params'][area_pre] = p['original']['withinarea'].copy()
        p['network_params'][area_pre]['N_total'] = int(rng.normal(p['original']['withinarea']['N_total'], p['unbalanced_network_sigma']*p['original']['withinarea']['N_total'] ))
        while p['network_params'][area_pre]['N_total'] <= p['original']['withinarea']['indegree']:
            p['network_params'][area_pre]['N_total'] = int(rng.normal(p['original']['withinarea']['N_total'], p['unbalanced_network_sigma']*p['original']['withinarea']['N_total'] ))
        number_neurons = p['network_params'][area_pre]['N_total']
        print(f'area {area_pre} has size {number_neurons}')
        for area_post in p['areas_list']:
            if area_pre != area_post:
                p['network_params'][area_pre][area_post] = p['original']['interareal'].copy()
    return p


