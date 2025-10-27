import GlitchPop as gp
import GlitchPop.simulate
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
from scipy.signal import resample_poly

def save_glitches(WnoiseL1, WglitchL1, WnoiseH1, WglitchH1):
    try:
        hf.close()
    except:
        pass

    hf = h5py.File('For_ARCHGEM2.hdf5', 'w')
    g1 = hf.create_group('injection_samples')
    g2 = hf.create_group('injection_parameters')

    g1.create_dataset('l1_strain', data=WnoiseL1)
    g2.create_dataset('l1_signal_whitened', data=WglitchL1)

    g1.create_dataset('h1_strain', data=WnoiseH1)
    g2.create_dataset('h1_signal_whitened', data=WglitchH1)

    hf.close()

def get_tds(ifo='L1', run='O4a', duration=50, seed=69):
    td = gp.simulate.determ_glitch(ifo, run, glitches = [], times = [], seeds = [seed], duration=duration, gps=0, scale=0.01)[0] # no glitches just noise
    td_empty = gp.simulate.determ_glitch(ifo, run, glitches = [], times = [], seeds = [seed], duration=duration, gps=0, scale=0)[0] # literally zeros
    ASD =  gp.simulate.determ_glitch(ifo, run, glitches = [], times = [], seeds = [seed], duration=201, gps=0, scale=0.01)[0].asd() #Generates an ASD for Gaussian noise
    return td, td_empty, ASD

def only_tds(ifo='L1', run='O4a', duration=50, seed=69):
    td = gp.simulate.determ_glitch(ifo, run, glitches = [], times = [], seeds = [seed], duration=duration, gps=0, scale=0.01)[0] # no glitches just noise
    td_empty = gp.simulate.determ_glitch(ifo, run, glitches = [], times = [], seeds = [seed], duration=duration, gps=0, scale=0)[0] # literally zeros
    return td, td_empty
    
def generate_scatter(params):
    duration, n_harmonics , harmonic_frequency_delta, base_harmonic_frequency, modulation_frequency, phi, amplitude = params
    scat = gp.scatter.slow_scattering(duration, n_harmonics , harmonic_frequency_delta, base_harmonic_frequency, modulation_frequency, phi, amplitude)
    return scat

def add_signals(td, glitch1_offset, glitch2_offset, gwoffset, ifo='L1', gw_type='BHBH', duration=50, seed=10, gseed=2, N=10):

    try:
        params1 = func_timeout(20, gp.scatter.slow_params, kwargs={'seed': gseed})
    except:
        params1 = func_timeout(30, gp.scatter.slow_params, kwargs={'seed': gseed+2*N})
    scat1 = generate_scatter(params1)

    try:
        params2 = func_timeout(20, gp.scatter.slow_params, kwargs={'seed': gseed+(N+1)})
    except:
        params2 = func_timeout(30, gp.scatter.slow_params, kwargs={'seed': gseed+(N+2*N)})
    scat2 = generate_scatter(params2)
    custom_params = {
        "custom_event1": {
            "time": duration/2+gwoffset,     # must match the merger time in t_merge for that signal
            "f_lower": 60    # your desired start frequency (Hz)
        }
    }

    td, gw_params = gp.simulate.determ_gw(ifo=ifo, ts=td, signals=[gw_type], t_merge=[duration/2+gwoffset], seeds=[seed], approximants=['IMRPhenomD'], custom_params=custom_params) #IMRPhenomD_NRTidal
    td = gp.scatter.inject_glitch(td, scat1, duration/2+glitch1_offset)
    td = gp.scatter.inject_glitch(td, scat2, duration/2+glitch2_offset)
    return td, [gw_params, params1, params2]

N = 41
glitch1_offset = np.random.uniform(-1, 0.5, N)
glitch2_offset = np.random.uniform(4, 6, N)
gwoffset = np.random.uniform(-6, -3, N)
gw_type = ['BHBH', 'NSBH', 'BNS']

duration = 20
ASDL1 = get_tds(ifo='L1')[2]
ASDH1 = get_tds(ifo='H1')[2]

Params = []

wtdL1 = np.zeros((N,duration*2048+1),dtype=np.float32)
wtdL1_empty = np.zeros((N,duration*2048+1), dtype=np.float32)

wtdH1 = np.zeros((N,duration*2048+1),dtype=np.float32)
wtdH1_empty = np.zeros((N,duration*2048+1), dtype=np.float32)

for ind in tqdm(range(N)):
    td, td_empty = only_tds(seed=ind, duration=duration)
    td, params = add_signals(td, glitch1_offset[ind], glitch2_offset[ind], gwoffset[ind], seed=ind, gseed=ind, gw_type=gw_type[0], N=N, duration=duration)
    td_empty = add_signals(td_empty, glitch1_offset[ind], glitch2_offset[ind], gwoffset[ind], seed=ind, gseed=ind, gw_type=gw_type[0], N=N, duration=duration)[0]
    Params.append(params)

    wtdL1[ind] = resample_poly(td.whiten(asd=ASDL1), up=1, down=4).astype('float32')
    wtdL1_empty[ind] = resample_poly(td_empty.whiten(asd=ASDL1), up=1, down=4).astype('float32')

    td, td_empty = only_tds(seed=ind, ifo='H1', duration=duration)
    td = add_signals(td, glitch1_offset[ind], glitch2_offset[ind], gwoffset[ind], seed=ind, gseed=ind, gw_type=gw_type[0], N=N, ifo='H1', duration=duration)[0]
    td_empty = add_signals(td_empty, glitch1_offset[ind], glitch2_offset[ind], gwoffset[ind], seed=ind, gseed=ind, gw_type=gw_type[0], N=N, ifo='H1', duration=duration)[0]

    wtdH1[ind] = resample_poly(td.whiten(asd=ASDH1), up=1, down=4).astype('float32')
    wtdH1_empty[ind] = resample_poly(td_empty.whiten(asd=ASDH1), up=1, down=4).astype('float32')

save_glitches(wtdL1, wtdL1_empty, wtdH1, wtdH1_empty)

ps = np.array(Params, dtype=object)
np.save('GW_params.npy', ps[:,0])
np.save('Scatter_glitch1_params.npy', ps[:,1])
np.save('Scatter_glitch2_params.npy', ps[:,2])
