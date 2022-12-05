import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
#import elephant


def lambertwm1(x):
    """Wrapper for LambertWm1 function"""
    # Using scipy to mimic the gsl_sf_lambert_Wm1 function.
    return sp.lambertw(x, k=-1 if x < 0 else 0).real

def convert_synapse_weight(tau_m, tau_syn, C_m):
    """
    Computes conversion factor for synapse weight from mV to pA
    This function is specific to the leaky integrate-and-fire neuron
    model with exp-shaped postsynaptic currents.
    """

    # compute time to maximum of V_m after spike input
    # to neuron at rest

    sub = 1. / (tau_syn - tau_m)
    pre = tau_m * tau_syn / C_m * sub
    frac = (tau_m / tau_syn) ** sub

    PSC_over_PSP = 1. / (pre * (frac**tau_m - frac**tau_syn))
    return PSC_over_PSP

def adjust_weights_and_input_to_synapse_scaling(params, conv_factor):
    conv_factor_new = conv_factor/np.sqrt(params['scale'])
    return conv_factor_new

def compute_rate(Nrec, simtime, sr):
    """Compute local approximation of average firing rate
    This approximation is based on the number of local nodes, number
    of local spikes and total time. Since this also considers devices,
    the actual firing rate is usually underestimated.
    """

    n_local_spikes = sr.n_events
    times = sr.get('events')['times']
    print('rate = ', n_local_spikes, ' / (', Nrec, ' * ', simtime, ' * 1e3 )')
    return 1. * n_local_spikes / (Nrec * simtime) * 1e3

def compute_fano_factor(sr):
    senders = sr.get('events')['senders']
    times = sr.get('events')['times']

    times_per_sender = []
    for sender in senders:
        if sender == 0:
            break
        idx, = np.where(senders==sender)
        senders[idx] == 0
        times_per_sender.append(times[idx])

    f = elephant.statistics.fanofactor(times_per_sender)
    return f

def plot_spikes_hist(sr, simtime):
    times = sr.get('events')['times']

    bins = int(simtime//10)

    print('Plotting histogram.')

    plt.hist(times, bins=bins, alpha=0.5, label='spikes Exc neurons')
    plt.xlim([0, simtime])
    plt.legend()
    plt.xlabel('time [ms]')
    plt.ylabel('# spikes')
    # plt.show()
    plt.savefig('spikes_hist.png')

