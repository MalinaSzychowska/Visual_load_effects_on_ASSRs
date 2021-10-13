import os, mne, sys
import os.path as op
import numpy as np
import numpy.fft as fft
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import glob
from mne.viz import plot_compare_evokeds as pce

# Stefan
#dir_main = op.join(op.expanduser("~"),'Documents','Box Sync', 'ASSR_study2')
# Malina
dir_main = op.join(op.expanduser("~"),'Box Sync', 'ASSR_study2')
# EEG lab
#dir_main = op.join(op.expanduser("~"), 'Documents', 'Malina_Box','Box Sync', 'ASSR_study2') # EEG lab

sys.path.append(dir_main + '\\MNE')

#mne.set_log_level('WARNING')

def ERP(c, evodict):
    '''
    take the individual ERP data for a single condition and return a list
    note that list(evodict[c]) gives all subject numbers in a list
    Arguments:
        c -- key from evodict dictionary (experimental condition)
        evodict -- dictionary with experimental conditions, each key in evodict 
            is a dictionary itself, with participant's number (key) 
            and evoked arrays (value)
    '''
    return [evodict[c][i] for i in list(evodict[c])]

def extract_mean_amps(d, t, ch, fp, cond):
    '''
    extract mean amplitudes across time and channels
    saved in V, convert to microV
    Arguments:
        d -- dictionary with experimental conditions, each key in evodict 
            is a dictionary itself, with participant's number (key) 
            and evoked arrays (value)
        t -- samples from which to extract the mean amps 
        ch -- channels from which to extract the mean amps
        fp -- participant number
        cond -- experimental condition (must match one of the keys in d)
    '''
    return np.mean([dx[t] for dx in d[cond][fp].data[ch]]) * 1e6

# Set file paths
dir_results = op.join(dir_main, 'results')
dir_evo = op.join(dir_main, 'MNE', 'fif', 'evokeds')
dir_epo = op.join(dir_main, 'MNE', 'fif', 'epochs')

# ========================
# Process auditory data
# ========================
# data are saved as epochs
# load filenames
os.chdir(dir_epo)  # change directory to averages folder
files = glob.glob('*audio-epo.fif')

# same condition information as in preprocess file
conds = {'no1': 11, 'no2': 21,'no3': 31,'no4': 41,
         'lo1': 12, 'lo2': 22,'lo3': 32,'lo4': 42,
         'hi1': 13, 'hi2': 23,'hi3': 33,'hi4': 43,
         'vh1': 14, 'vh2': 24,'vh3': 34,'vh4': 44}
conds = list(conds)

# We have these electrodes:
# ['Nz', 'Fpz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'Cheek', 'EOG']
# Use Fz and FCz for audio
chanpick = ['Fz', 'FCz']

# save evoked (i.e., averages)
evo_audio = {i:{} for i in conds}

fps = []
# save data by subject (for export)
Dall = [] # save all channels
Dsel = [] # save only signal and noise channels
# save data by condition (for figures)
Damp = {i:[] for i in conds}
Ditc = {i:[] for i in conds}
for f in files:
    # f = files[0]
    print('\n'+'='*50)
    fp = int(f[2:4])
    print('Subject:', fp)
    print('='*50)
    fps += [fp]
    epo_fp = mne.read_epochs(f, preload=True)
    tmpall = {'fp': fp} # save all channels
    tmpsel = {'fp': fp} # save only signal and noise channels
    for cond in conds:
        #cond = 'no1'
        # compute GM (ie evokeds)
        # =========================
        ev = epo_fp[cond].average()
        #ev.set_eeg_reference(ref_channels=['Nz'])
        ev.info['subject_info'] = fp
        print(cond + " = " + str(ev.nave) + " trials")
        evo_audio[cond][fp] = ev
        chans = [ev.ch_names.index(c) for c in chanpick]

        # use fft from numpy
        # ==================
        # select frequencies (by defining min and max below)
        Fs = ev.info['sfreq']
        fftfreq = fft.fftfreq(len(ev.times), 1/Fs)
        myfreqs = fftfreq[np.logical_and(fftfreq > 25, fftfreq < 55)]
        #[25.6  26.24 26.88 27.52 28.16 28.8  29.44 30.08 30.72 31.36 32.   32.64
        # 33.28 33.92 34.56 35.2  35.84 36.48 37.12 37.76 38.4  39.04 39.68 40.32
        # 40.96 41.6  42.24 42.88 43.52 44.16 44.8  45.44 46.08 46.72 47.36 48.
        # 48.64 49.28 49.92 50.56 51.2  51.84 52.48 53.12 53.76 54.4 ]
        freqsindx = np.where(np.isin(fftfreq,myfreqs))[0]

        # index signal and noise frequencies
        ASSRfreq = 40.96
        indx_signal = int(np.where(myfreqs == ASSRfreq)[0])
        # how many neighboring freqs for the SNR analysis?
        nSNRfreqs = 10
        sur_lo = list(range(indx_signal-2-nSNRfreqs, indx_signal-2, 1))
        #for 10: [33.28 33.92 34.56 35.2  35.84 36.48 37.12 37.76 38.4  39.04]
        sur_hi = list(range(indx_signal+3, indx_signal+3+nSNRfreqs, 1))
        #for 10: [42.88 43.52 44.16 44.8  45.44 46.08 46.72 47.36 48.   48.64]
        indx_noise = sur_lo + sur_hi

        # extract amplitude spectrum for GM (ie evokeds)
        fftamps = np.abs(fft.fft(ev.data[chans], axis = 1))*1e3
        fftamps = fftamps[:,freqsindx]
        Damp[cond].append(np.mean(fftamps, axis = 0))

        for c in range(0,len(chanpick)):
            for i in range(0,len(myfreqs)):
                tmpall['Amp%s_%s_%.2f'%(cond, chanpick[c],myfreqs[i])] = fftamps[c,i]

        # S, N, SNR, and SmN for all channels
        tmpS = fftamps[:,indx_signal]
        tmpN = np.mean(fftamps[:,indx_noise],axis=1)
        tmpSNR = tmpS/tmpN
        tmpS = np.mean(tmpS)
        tmpN = np.mean(tmpN)
        tmpSNR = np.mean(tmpSNR)
        tmpsel['AmpS'+cond] = tmpS
        tmpsel['AmpN'+cond] = tmpN
        tmpsel['AmpSNR'+cond] = tmpSNR
        tmpsel['AmpSmN'+cond] = tmpS - tmpN

        # extract itc from individual trials (ie epocheds)
        # =================================================
        # fft results for selected channels
        F = fft.fft(epo_fp[cond].get_data()[:,chans,:], axis=2)
        # the number of epochs
        N = len(F)
        # some computations ...
        fftitc = F/np.abs(F)
        fftitc = np.sum(fftitc, 0)
        fftitc = np.abs(fftitc)/N

        # extract selected channels and freq of interest
        fftitc = fftitc[:,freqsindx]
        Ditc[cond].append(np.mean(fftitc, axis = 0))
        
        for c in range(0,len(chanpick)):
            for i in range(0,len(myfreqs)):
                tmpall['Itc%s_%s_%.2f'%(cond, chanpick[c],myfreqs[i])] = fftitc[c,i]

        # S, N, SNR, and SmN for all channels
        tmpS = fftitc[:,indx_signal]
        tmpN = np.mean(fftitc[:,indx_noise],axis=1)
        tmpSNR = tmpS/tmpN
        tmpS = np.mean(tmpS)
        tmpN = np.mean(tmpN)
        tmpSNR = np.mean(tmpSNR)
        tmpsel['ItcS'+cond] = tmpS
        tmpsel['ItcN'+cond] = tmpN
        tmpsel['ItcSNR'+cond] = tmpSNR
        tmpsel['ItcSmN'+cond] = tmpS - tmpN
    Dall += [tmpall]
    Dsel += [tmpsel]
for i in Damp:
    Damp[i] = np.row_stack(Damp[i])
for i in Ditc:
    Ditc[i] = np.row_stack(Ditc[i])
df = pd.DataFrame(Dall)
df.to_csv(op.join(dir_results, 'data_audio_all_channels.csv'), sep='\t', index=False)
df = pd.DataFrame(Dsel)
df.to_csv(op.join(dir_results, 'data_audio_signal_noise.csv'), sep='\t', index=False)
print('Done with ', len(fps), ' subjects:\n',fps)

# compute grand means across blocks
# =================================
evo_audioGM = {'no': {}, 'lo': {}, 'hi': {}, 'vh': {}}
for fp in fps:
    evo_audioGM['no'][fp] = mne.combine_evoked([evo_audio[i][fp] for i in conds[0:4]],
               weights = [0.25, 0.25, 0.25, 0.25])
    evo_audioGM['lo'][fp] = mne.combine_evoked([evo_audio[i][fp] for i in conds[4:8]],
               weights = [0.25, 0.25, 0.25, 0.25])
    evo_audioGM['hi'][fp] = mne.combine_evoked([evo_audio[i][fp] for i in conds[8:12]],
               weights = [0.25, 0.25, 0.25, 0.25])
    evo_audioGM['vh'][fp] = mne.combine_evoked([evo_audio[i][fp] for i in conds[12:16]],
               weights = [0.25, 0.25, 0.25, 0.25])
    # when I chose weights = 'equal', the weights were +1 each and the result was a sum of the ERPs
    # but we want averages


# plot mean amps and amplitude spectra for individual subjects
# ============================================================
#ylim = [-3, 3] # force a particular Y min and max
style = dict(colors={'No': 'black', 'Low': 'blue', 'High': 'orange', 'Very High': 'red'},
    ci=None, show_sensors=False, legend='upper right',
    linestyles={'No': '-', 'Low': '-', 'High': '-', 'Very High': '-'},
    truncate_xaxis=False, truncate_yaxis=False) #, ylim=dict(eeg=ylim))
for fp in fps:
    # fp = 1
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    pce({'No':  evo_audioGM['no'][fp],
         'Low': evo_audioGM['lo'][fp],
         'High': evo_audioGM['hi'][fp],
         'Very High': evo_audioGM['vh'][fp]},
         **style, picks=chans, combine = 'mean',
         axes = ax[0])
    fpind = fps.index(fp)
    # for each load, combine the four blocks
    tmpno = np.mean(np.row_stack([Damp[c][fps.index(fp)] for c in conds[0:4]]), axis=0)
    tmplo = np.mean(np.row_stack([Damp[c][fps.index(fp)] for c in conds[4:8]]), axis=0)
    tmphi = np.mean(np.row_stack([Damp[c][fps.index(fp)] for c in conds[8:12]]), axis=0)
    tmpvh = np.mean(np.row_stack([Damp[c][fps.index(fp)] for c in conds[12:16]]), axis=0)
    ax[1].plot(myfreqs, tmpno, 'k-', label='No')
    ax[1].plot(myfreqs, tmplo, 'b-', label='Low')
    ax[1].plot(myfreqs, tmphi, '-', color = 'orange', label='High')
    ax[1].plot(myfreqs, tmpvh, 'r-', label='Very High')
    ax[1].legend(frameon=False)
    ax[1].set_xlabel('Frequencies (Hz)')
    ax[1].set_ylabel('Amplitudes (µV)')
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].set_title("Amplitude spectrum")
    fig.suptitle("Subject "+str(fp))
    fig.tight_layout()
    plt.savefig(op.join(dir_results, 'figures', 'individual_aspctm', 'fp_%i.pdf' % fp), format='pdf', dpi = 600)
    plt.close(fig=None)


# plot grand mean amplitude spectrum by load
# ==========================================
plt.close(fig=None)
fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharey=False)
style = dict(show_sensors=False, legend='upper right',
             truncate_xaxis=False, truncate_yaxis=False, ci=None)
pce({'No': ERP('no', evo_audioGM),
     'Low': ERP('lo', evo_audioGM),
     'High': ERP('hi', evo_audioGM),
     'Very High': ERP('vh', evo_audioGM)},
     colors={'No': 'black', 'Low': 'blue', 'High': 'orange', 'Very High': 'red'},
     linestyles={'No': '-', 'Low': '-', 'High': '-', 'Very High': '-'},
     **style, title=", ".join(chanpick), picks=chans, combine = 'mean',
     axes = ax[0])
    #ylim=dict(eeg=[-6, 12])
ax[0].set_title('Mean ERP (top) and amplitude spectrum (bottom) (N = %i)' % len(fps))
ax[0].set_ylabel('Amplitudes (µV)')
ax[0].legend(frameon=False)
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# for each load, combine the four blocks
tmpno = np.mean(np.row_stack([np.row_stack([Damp[c][fps.index(fp)] for c in conds[0:4]]) for fp in fps]), axis=0)
tmplo = np.mean(np.row_stack([np.row_stack([Damp[c][fps.index(fp)] for c in conds[4:8]]) for fp in fps]), axis=0)
tmphi = np.mean(np.row_stack([np.row_stack([Damp[c][fps.index(fp)] for c in conds[8:12]]) for fp in fps]), axis=0)
tmpvh = np.mean(np.row_stack([np.row_stack([Damp[c][fps.index(fp)] for c in conds[12:16]]) for fp in fps]), axis=0)
ax[1].plot(myfreqs, tmpno, 'k-', label='No')
ax[1].plot(myfreqs, tmplo, 'b-', label='Low')
ax[1].plot(myfreqs, tmphi, '-', color = 'orange', label='High')
ax[1].plot(myfreqs, tmpvh, 'r-', label='Very High')
ax[1].legend(frameon=False)
ax[1].set_xlabel('Frequencies (Hz)')
ax[1].set_ylabel('Amplitudes (µV)')
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.tight_layout()
fig.savefig(op.join(dir_results, 'figures', 'figure_mean_ERP_aspctm.pdf'), format='pdf', dpi = 600)


# plot grand mean itc by load
# ===========================
plt.close(fig=None)
fig, ax = plt.subplots(figsize=(14, 9), sharey=False) #subplot seems to be needed to define spines below
# for each load, combine the four blocks
tmpno = np.mean(np.row_stack([np.row_stack([Ditc[c][fps.index(fp)] for c in conds[0:4]]) for fp in fps]), axis=0)
tmplo = np.mean(np.row_stack([np.row_stack([Ditc[c][fps.index(fp)] for c in conds[4:8]]) for fp in fps]), axis=0)
tmphi = np.mean(np.row_stack([np.row_stack([Ditc[c][fps.index(fp)] for c in conds[8:12]]) for fp in fps]), axis=0)
tmpvh = np.mean(np.row_stack([np.row_stack([Ditc[c][fps.index(fp)] for c in conds[12:16]]) for fp in fps]), axis=0)
plt.plot(myfreqs, tmpno, 'k-', label='No')
plt.plot(myfreqs, tmplo, 'b-', label='Low')
plt.plot(myfreqs, tmphi, '-', color = 'orange', label='High')
plt.plot(myfreqs, tmpvh, 'r-', label='Very High')
plt.title('Intertrial coherence (itc)')
plt.legend(frameon=False)
plt.xlabel('Frequencies (Hz)')
plt.ylabel('itc')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(op.join(dir_results, 'figures', 'figure_audio_itc.pdf'), format='pdf', dpi = 600)
plt.close(fig=None)


# ========================
# process visual data
# ========================

# load filenames
os.chdir(dir_evo)  # switch to averages folder
files = glob.glob('*_visual_Nonno1-ave.fif')

# same condition information as in preprocess file
conds = {'Nonno1': 11, 'Nonno2': 12, 'Nonno3': 13, 'Nonno4': 14,
         'Nonlo1': 21, 'Nonlo2': 22, 'Nonlo3': 23, 'Nonlo4': 24,
         'Tarlo1': 31, 'Tarlo2': 32, 'Tarlo3': 33, 'Tarlo4': 34,
         'Nonhi1': 41, 'Nonhi2': 42, 'Nonhi3': 43, 'Nonhi4': 44,
         'Tarhi1': 51, 'Tarhi2': 52, 'Tarhi3': 53, 'Tarhi4': 54,
         'Nonvh1': 61, 'Nonvh2': 62, 'Nonvh3': 63, 'Nonvh4': 64,
         'Tarvh1': 71, 'Tarvh2': 72, 'Tarvh3': 73, 'Tarvh4': 74}
conds = list(conds)


# compute ERPs per condition per subject
# ======================================
evo_visual = {i:{} for i in conds}
evo_visual_lp = {i:{} for i in conds}
fps = []
for f in files:
    # f = files[0]
    print('\n'+'='*50)
    fp = int(f[2:4])
    print('Subject:', fp)
    print('='*50)
    fps += [fp]
    for c in conds:
        ev = mne.read_evokeds(op.join(dir_main, 'MNE', 'fif', 'evokeds',
                                      'fp%02d_visual_' % fp + c + '-ave.fif' ), condition = c)
        print(ev.comment+" = "+str(ev.nave))
        ev.info['subject_info'] = fp
        ev.set_eeg_reference(ref_channels=['Nz'])
        ev.apply_baseline(baseline=(None, 0)) # from beginning of interval to onset; so, from -100 to 0
        ev_lp = ev.copy() #create a copy for low pass
        ev_lp.filter(h_freq=30, l_freq=None)
        evo_visual[ev.comment][fp] = ev
        evo_visual_lp[ev_lp.comment][fp] = ev_lp

# compute grand means across blocks by load and target/nontarget
# ==============================================================
evo_visualGM = {'Tarlo': {},'Tarhi': {}, 'Tarvh': {},
                'Nonno': {}, 'Nonlo': {},'Nonhi': {}, 'Nonvh': {}}
for fp in fps:
    evo_visualGM['Tarlo'][fp] = mne.combine_evoked([evo_visual_lp[conds[i]][fp]
        for i in np.where(['Tarlo' in c for c in conds])[0]], weights = [0.25, 0.25, 0.25, 0.25])
    evo_visualGM['Tarhi'][fp] = mne.combine_evoked([evo_visual_lp[conds[i]][fp]
        for i in np.where(['Tarhi' in c for c in conds])[0]], weights = [0.25, 0.25, 0.25, 0.25])
    evo_visualGM['Tarvh'][fp] = mne.combine_evoked([evo_visual_lp[conds[i]][fp]
        for i in np.where(['Tarvh' in c for c in conds])[0]], weights = [0.25, 0.25, 0.25, 0.25])
    evo_visualGM['Nonno'][fp] = mne.combine_evoked([evo_visual_lp[conds[i]][fp]
        for i in np.where(['Nonno' in c for c in conds])[0]], weights = [0.25, 0.25, 0.25, 0.25])
    evo_visualGM['Nonlo'][fp] = mne.combine_evoked([evo_visual_lp[conds[i]][fp]
        for i in np.where(['Nonlo' in c for c in conds])[0]], weights = [0.25, 0.25, 0.25, 0.25])
    evo_visualGM['Nonhi'][fp] = mne.combine_evoked([evo_visual_lp[conds[i]][fp]
        for i in np.where(['Nonhi' in c for c in conds])[0]], weights = [0.25, 0.25, 0.25, 0.25])
    evo_visualGM['Nonvh'][fp] = mne.combine_evoked([evo_visual_lp[conds[i]][fp]
        for i in np.where(['Nonvh' in c for c in conds])[0]], weights = [0.25, 0.25, 0.25, 0.25])
    # when I chose weights = 'equal', the weights were +1 each and the result was a sum of the ERPs
    # but we want averages


# plot grand means across blocks by load and target/nontarget
# ==============================================================
# We have these electrodes:
# keepchannels = ['Nz', 'Fpz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'Cheek', 'EOG']
# Use Cz, PCz, and Pz for P3
chanpick = ['Cz', 'CPz', 'Pz']
chans = [evo_visualGM['Tarlo'][1].ch_names.index(c) for c in chanpick]

plt.close(fig=None)
style = dict(show_sensors=False, legend='lower center',
             truncate_xaxis=False, truncate_yaxis=False, ci=None)
pce({'Low Targets': ERP('Tarlo', evo_visualGM),
     'High Targets': ERP('Tarhi', evo_visualGM),
     'Very High Targets': ERP('Tarvh', evo_visualGM),
     'Low Non-targets': ERP('Nonlo', evo_visualGM),
     'High Non-targets': ERP('Nonhi', evo_visualGM),
     'Very High Non-targets': ERP('Nonvh', evo_visualGM)},
     #'No load': ERP('Nonno', evo_visualGM)},
     colors={'Low Targets': 'blue', 'Low Non-targets': 'blue', #'No load': 'black',
             'High Targets': 'orange', 'High Non-targets': 'orange',
             'Very High Targets': 'red', 'Very High Non-targets': 'red'},
     linestyles={'Low Targets': '-', 'Low Non-targets': ':', #'No load': '-',
                 'High Targets': '-', 'High Non-targets': ':',
                 'Very High Targets': '-', 'Very High Non-targets': ':'},
     **style, title=", ".join(chanpick),
     picks=chans, combine = 'mean') #ylim=dict(eeg=[-5, 11])
    # if you have more than one channel, you need to have: combine = 'mean'
    # otherwise, mne computes GFP
plt.tight_layout()
# save figure
plt.savefig(op.join(dir_results, 'figures', 'figure_visual_ERP.pdf'), format='pdf', dpi = 600)
plt.close(fig=None)


# extract mean amplitudes
# =======================
erp = evo_visual
P3int = np.where((erp[list(erp)[0]][1].times >= 0.300) & (erp[list(erp)[0]][1].times <= 0.400))
chanpick = ['Cz', 'CPz', 'Pz']
chans = [erp[list(erp)[0]][1].ch_names.index(c) for c in chanpick]
x = []
conds = list(erp)
for fp in fps:
    tmp = {'fp': fp}
    for i in conds:
        #tmp['Ntrials_'+i] = erp[i][fp].nave
        tmp['vP3_'+i] = extract_mean_amps(erp, P3int, chans, fp, i)
    x += [tmp]
df = pd.DataFrame(x)
cols = ['fp'] + ['vP3_'+i for i in conds]
#cols = ['fp'] + ['P3_'+i for i in conds] + ['Ntrials_'+i for i in conds]
df = df.reindex(columns=cols)
df.to_csv(op.join(dir_results, 'data_visualP3.csv'), sep='\t', index=False)
