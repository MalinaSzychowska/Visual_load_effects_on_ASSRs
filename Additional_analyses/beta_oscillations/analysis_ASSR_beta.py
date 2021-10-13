import os, mne
import os.path as op
import numpy as np
import numpy.fft as fft
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import glob
from mne.viz import plot_compare_evokeds as pce


# Stefan
#dir_main = op.join(op.expanduser("~"),'Documents','Box Sync', 'ASSR2')
# Malina
dir_main = op.join(op.expanduser("~"),'Box Sync') #, 'ASSR2')
# EEG lab
#dir_main = op.join(op.expanduser("~"), 'Documents', 'Malina_Box','Box Sync', 'ASSR2') # EEG lab

dir_results = op.join(dir_main, 'Malina_dissertation', 'additional analyses for the defense', 'beta oscillations', 'results')
#sys.path.append(dir_main + '\\MNE')

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


# Settings for different studies

studies = ['ASSR_study1','ASSR_study2','ASSR2']

conds_ASSR_study1 = {'lo1': 1, 'lo2': 2,'lo3': 3,'lo4': 4,
                     'hi1': 5, 'hi2': 6,'hi3': 7,'hi4': 8}

conds_ASSR_study2 = {'no1': 11, 'no2': 21,'no3': 31,'no4': 41,
                     'lo1': 12, 'lo2': 22,'lo3': 32,'lo4': 42,
                     'hi1': 13, 'hi2': 23,'hi3': 33,'hi4': 43,
                     'vh1': 14, 'vh2': 24,'vh3': 34,'vh4': 44}

conds_ASSR2 = {'no20_1': 121, 'no20_2': 122,
              'no40_1': 141, 'no40_2': 142,
              'no80_1': 181, 'no80_2': 182,
              'lo20_1': 221, 'lo20_2': 222,
              'lo40_1': 241, 'lo40_2': 242,
              'lo80_1': 281, 'lo80_2': 282,
              'hi20_1': 321, 'hi20_2': 322,
              'hi40_1': 341, 'hi40_2': 342,
              'hi80_1': 381, 'hi80_2': 382}

conds_all = [list(conds_ASSR_study1), list (conds_ASSR_study2), list(conds_ASSR2)]
evo_audio_allstudies = {i:{} for i in studies}
Damp_allstudies = {i:[] for i in studies} # amplitudes for the selected frequency spectrum
Ditc_allstudies = {i:[] for i in studies} # intertrial coherence for selected freq spectrum
fps_allstudies= {i:[] for i in studies}

for study, conds in zip(studies, conds_all): 
    print('\n'+'='*50)
    print('Going through:', study)
    print('='*50)
    #study = 'ASSR_study1'   
    #conds = list(conds_ASSR_study1)
    dir_evo = op.join(dir_main, study, 'MNE', 'fif', 'evokeds')
    dir_epo = op.join(dir_main, study, 'MNE', 'fif', 'epochs')
    
    # ========================
    # Process auditory data
    # ========================
    # data are saved as epochs
    # Load filenames
    os.chdir(dir_epo)  # change directory to averages folder
    files = glob.glob('*audio-epo.fif')

    # Use Fz and FCz for audio
    chanpick = ['Fz', 'FCz']
    
    # save evoked (i.e., averages)
    evo_audio = {i:{} for i in conds}
    
    fps = []
    # save data by subject (for export)
    Dall = [] # save all channels
    Dsel = [] # save only signal and noise channels
    # save data by condition (for figures)
    Damp_all = {i:[] for i in conds} # amplitudes for the wide frequency spectrum
    Damp = {i:[] for i in conds} # amplitudes for the selected frequency spectrum
    Damp_indEpo = {i:[] for i in conds} # amplitudes when first FFT and then avg
    Ditc = {i:[] for i in conds} # intertrial coherence for selected freq spectrum
    
    ## if you only want to run the participants after preregistration:
    # files = files[8:len(files)]
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
            # cond = conds[0]
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

            ASSRfreq = 16
            
            myfreqs_all = fftfreq[9:160] # wide freq spectrum, up to  bit more than 100 Hz
            freqsindx_all = np.where(np.isin(fftfreq,myfreqs_all))[0]
    
            myfreqs = fftfreq[np.logical_and(fftfreq > ASSRfreq-15, fftfreq < ASSRfreq+15)]
            freqsindx = np.where(np.isin(fftfreq,myfreqs))[0]
    
            # index signal and noise frequencies
            indx_signal = int(np.where(myfreqs == ASSRfreq)[0])
            # how many neighboring noise frequencies?
            nSNRfreqs = 8
            sur_lo = list(range(indx_signal-2-nSNRfreqs, indx_signal-2, 1))
            sur_hi = list(range(indx_signal+3, indx_signal+3+nSNRfreqs, 1))
            indx_noise = sur_lo + sur_hi
            # myfreqs[indx_noise]
           
            # extract amplitude spectrum for GM (ie evokeds)
            fftamps = np.abs(fft.fft(ev.data[chans], axis = 1))*1e3
            fftamps_all = fftamps[:,freqsindx_all]
            fftamps = fftamps[:,freqsindx]
            
            fft_indEpo = np.mean(np.abs(fft.fft(epo_fp[cond].get_data()[:,chans,:], axis = 1))*1e3, axis = 0)
            fft_indEpo = fft_indEpo[:,freqsindx_all]
    
            Damp[cond].append(np.mean(fftamps, axis = 0))
            Damp_all[cond].append(np.mean(fftamps_all, axis = 0))
            Damp_indEpo[cond].append(np.mean(fft_indEpo, axis = 0))
    
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
                    tmpall['Itc%s_%.2f'%(chanpick[c],myfreqs[i])] = fftitc[c,i]
    
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
    for i in Damp_all:
        Damp_all[i] = np.row_stack(Damp_all[i])
    for i in Damp_indEpo:
        Damp_indEpo[i] = np.row_stack(Damp_indEpo[i])
    for i in Ditc:
        Ditc[i] = np.row_stack(Ditc[i])
    
    evo_audio_allstudies[study] = evo_audio
    Damp_allstudies[study] = Damp # amplitudes for the selected frequency spectrum
    Ditc_allstudies[study] = Ditc # intertrial coherence for selected freq spectrum
    fps_allstudies[study] = fps
    
    df = pd.DataFrame(Dall)
    df.to_csv(op.join(dir_results, 'data_audio_all_channels_{}_beta.tsv'.format(study)), sep='\t', index=False)
    df = pd.DataFrame(Dsel)
    df.to_csv(op.join(dir_results, 'data_audio_signal_noise_{}_beta.tsv'.format(study)), sep='\t', index=False)
    print('Done with ', len(fps), ' subjects:\n',fps)


# compute grand means across blocks for separate studies:
# =============================================================================
# ASSR_study1:
#------------------------------------------------------------------------------
evo_audio = evo_audio_allstudies['ASSR_study1']
Damp = Damp_allstudies['ASSR_study1']
Ditc = Ditc_allstudies['ASSR_study1']
fps = fps_allstudies['ASSR_study1']
conds = list(conds_ASSR_study1)


# compute grand means across blocks
# =================================
evo_audioGM = {'lo': {}, 'hi': {}}
for fp in fps:
    evo_audioGM['lo'][fp] = mne.combine_evoked([evo_audio[i][fp] for i in conds[0:4]],
               weights = [0.25, 0.25, 0.25, 0.25])
    evo_audioGM['hi'][fp] = mne.combine_evoked([evo_audio[i][fp] for i in conds[4:8]],
               weights = [0.25, 0.25, 0.25, 0.25])
    # when I chose weights = 'equal', the weights were +1 each and the result was a sum of the ERPs
    # but we want averages


# plot mean amps and amplitude spectra for individual subjects
# ============================================================
#ylim = [-3, 3] # force a particular Y min and max
style = dict(colors={'Low': 'blue', 'High': 'red'},
    ci=None, show_sensors=False, legend='upper right',
    linestyles={'Low': '-', 'High': '-'},
    truncate_xaxis=False, truncate_yaxis=False) #, ylim=dict(eeg=ylim))
plt.rcParams.update({'font.size': 14})

for fp in fps:
    # fp = 1
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    pce({'Low':  evo_audioGM['lo'][fp],
         'High': evo_audioGM['hi'][fp]},
         **style, picks=chans, combine = 'mean',
         axes = ax[0])
    fpind = fps.index(fp)
    # for each load, combine the four blocks
    tmplo = np.mean(np.row_stack([Damp[c][fps.index(fp)] for c in conds[0:4]]), axis=0)
    tmphi = np.mean(np.row_stack([Damp[c][fps.index(fp)] for c in conds[4:8]]), axis=0)
    ax[1].plot(myfreqs, tmplo, 'b-', label='Low')
    ax[1].plot(myfreqs, tmphi, 'r-', label='High')
    #ax[1].plot(myfreqs, Damp['lo'][fps.index(fp)], 'b-', label='Low')
    #ax[1].plot(myfreqs, Damp['hi'][fps.index(fp)], 'r-', label='High')
    ax[1].legend(frameon=False)
    ax[1].set_xlabel('Frequencies (Hz)')
    ax[1].set_ylabel('Amplitudes (µV)')
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].set_ylim(bottom = 0.05, top = 0.3)
    ax[1].axvspan(min(myfreqs[indx_noise]), myfreqs[indx_noise[7]], color='gray', alpha=0.2, lw=0)
    ax[1].axvspan(myfreqs[indx_noise[8]], max(myfreqs[indx_noise]), color='gray', alpha=0.2, lw=0)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].set_title("Amplitude spectrum")
    fig.suptitle("Subject "+str(fp))
    fig.tight_layout()
    plt.savefig(op.join(dir_results, 'figures', 'individual_aspctm', 'fp_{}_beta_ASSR_study1.jpg'.format(fp)), format='jpg', dpi = 600)
    plt.savefig(op.join(dir_results, 'figures', 'individual_aspctm', 'fp_{}_beta_ASSR_study1.pdf'.format(fp)), format='pdf', dpi = 600)
    plt.close(fig=None)


# plot grand mean amplitude spectrum by load
# ==========================================
plt.close(fig=None)
fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharey=False)
style = dict(show_sensors=False, legend='upper right',
             truncate_xaxis=False, truncate_yaxis=False, ci=None)
pce({'Low':  ERP('lo', evo_audioGM),
     'High': ERP('hi', evo_audioGM)},
     colors={'Low': 'blue', 'High': 'red'},
     linestyles={'Low': '-', 'High': '-'},
     **style, title=", ".join(chanpick), picks=chans, combine = 'mean',
     axes = ax[0])
    #ylim=dict(eeg=[-6, 12])
ax[0].set_title('Mean ERP (top) and amplitude spectrum (bottom) (N = %i)' % len(fps))
ax[0].set_ylabel('Amplitudes (µV)')
ax[0].legend(frameon=False)
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# for each load, combine the four blocks
tmplo = np.mean(np.row_stack([np.row_stack([Damp[c][fps.index(fp)] for c in conds[0:4]]) for fp in fps]), axis=0)
tmphi = np.mean(np.row_stack([np.row_stack([Damp[c][fps.index(fp)] for c in conds[4:8]]) for fp in fps]), axis=0)
ax[1].plot(myfreqs, tmplo, 'b-', label='Low')
ax[1].plot(myfreqs, tmphi, 'r-', label='High')
ax[1].legend(frameon=False)
ax[1].set_xlabel('Frequencies (Hz)')
ax[1].set_ylabel('Amplitudes (µV)')
ax[1].axvspan(min(myfreqs[indx_noise]), myfreqs[indx_noise[7]], color='gray', alpha=0.2, lw=0)
ax[1].axvspan(myfreqs[indx_noise[8]], max(myfreqs[indx_noise]), color='gray', alpha=0.2, lw=0)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_ylim(bottom = 0.05, top = 0.3)
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.tight_layout()
fig.savefig(op.join(dir_results, 'figures', 'figure_average_aspctm_and_ERP_beta_ASSR_study1.jpg'), format='jpg', dpi=600)
fig.savefig(op.join(dir_results, 'figures', 'figure_average_aspctm_and_ERP_beta_ASSR_study1.pdf'), format='pdf', dpi=600)
plt.close(fig=None)

# plot grand mean itc by load
# ===========================
fig, ax = plt.subplots(figsize=(14, 9), sharey=False) #subplot seems to be needed to define spines below
# for each load, combine the four blocks
tmplo = np.mean(np.row_stack([np.row_stack([Ditc[c][fps.index(fp)] for c in conds[0:4]]) for fp in fps]), axis=0)
tmphi = np.mean(np.row_stack([np.row_stack([Ditc[c][fps.index(fp)] for c in conds[4:8]]) for fp in fps]), axis=0)
plt.plot(myfreqs, tmplo, 'b-', label='Low')
plt.plot(myfreqs, tmphi, 'r-', label='High')
plt.title('Intertrial coherence (itc)')
plt.legend(frameon=False)
plt.xlabel('Frequencies (Hz)')
plt.ylabel('itc')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(bottom = 0.05, top = 0.2)
ax.axvspan(min(myfreqs[indx_noise]), myfreqs[indx_noise[7]], color='gray', alpha=0.2, lw=0)
ax.axvspan(myfreqs[indx_noise[8]], max(myfreqs[indx_noise]), color='gray', alpha=0.2, lw=0)
fig.tight_layout()
fig.savefig(op.join(dir_results, 'figures', 'figure_audio_itc_beta_ASSR_study1.jpg'), format='jpg', dpi=600)
fig.savefig(op.join(dir_results, 'figures', 'figure_audio_itc_beta_ASSR_study1.pdf'), format='pdf', dpi=600)
plt.close(fig=None)






# ASSR_study2:
#------------------------------------------------------------------------------
evo_audio = evo_audio_allstudies['ASSR_study2']
Damp = Damp_allstudies['ASSR_study2']
Ditc = Ditc_allstudies['ASSR_study2']
fps = fps_allstudies['ASSR_study2']
conds = list(conds_ASSR_study2)


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
plt.rcParams.update({'font.size': 14})

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
    ax[1].axvspan(min(myfreqs[indx_noise]), myfreqs[indx_noise[7]], color='gray', alpha=0.2, lw=0)
    ax[1].axvspan(myfreqs[indx_noise[8]], max(myfreqs[indx_noise]), color='gray', alpha=0.2, lw=0)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].set_ylim(bottom = 0.05, top = 0.3)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].set_title("Amplitude spectrum")
    fig.suptitle("Subject "+str(fp))
    fig.tight_layout()
    plt.savefig(op.join(dir_results, 'figures', 'individual_aspctm', 'fp_{}_beta_ASSR_study2.jpg'.format(fp)), format='jpg', dpi = 600)
    plt.savefig(op.join(dir_results, 'figures', 'individual_aspctm', 'fp_{}_beta_ASSR_study2.pdf'.format(fp)), format='pdf', dpi = 600)
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
ax[1].axvspan(min(myfreqs[indx_noise]), myfreqs[indx_noise[7]], color='gray', alpha=0.2, lw=0)
ax[1].axvspan(myfreqs[indx_noise[8]], max(myfreqs[indx_noise]), color='gray', alpha=0.2, lw=0)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_ylim(bottom = 0.05, top = 0.3)
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.tight_layout()
fig.savefig(op.join(dir_results, 'figures', 'figure_average_aspctm_and_ERP_beta_ASSR_study2.jpg'), format='jpg', dpi=600)
fig.savefig(op.join(dir_results, 'figures', 'figure_average_aspctm_and_ERP_beta_ASSR_study2.pdf'), format='pdf', dpi=600)
plt.close(fig=None)

# plot grand mean itc by load
# ===========================
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
ax.set_ylim(bottom = 0.05, top = 0.2)
ax.axvspan(min(myfreqs[indx_noise]), myfreqs[indx_noise[7]], color='gray', alpha=0.2, lw=0)
ax.axvspan(myfreqs[indx_noise[8]], max(myfreqs[indx_noise]), color='gray', alpha=0.2, lw=0)
fig.tight_layout()
fig.savefig(op.join(dir_results, 'figures', 'figure_audio_itc_beta_ASSR_study2.jpg'), format='jpg', dpi=600)
fig.savefig(op.join(dir_results, 'figures', 'figure_audio_itc_beta_ASSR_study2.pdf'), format='pdf', dpi=600)
plt.close(fig=None)





# ASSR2:
#------------------------------------------------------------------------------
evo_audio = evo_audio_allstudies['ASSR2']
Damp = Damp_allstudies['ASSR2']
Ditc = Ditc_allstudies['ASSR2']
fps = fps_allstudies['ASSR2']
conds = list(conds_ASSR2)

evo_audioGM = {'no20': {}, 'no40': {},'no80': {},
               'lo20': {}, 'lo40': {},'lo80': {},
               'hi20': {}, 'hi40': {},'hi80': {}}


for fp in fps:
    evo_audioGM['no20'][fp] = mne.combine_evoked([evo_audio[i][fp] for i in conds[0:2]],weights = [0.5, 0.5])
    evo_audioGM['no40'][fp] = mne.combine_evoked([evo_audio[i][fp] for i in conds[2:4]],weights = [0.5, 0.5])
    evo_audioGM['no80'][fp] = mne.combine_evoked([evo_audio[i][fp] for i in conds[4:6]],weights = [0.5, 0.5])
    evo_audioGM['lo20'][fp] = mne.combine_evoked([evo_audio[i][fp] for i in conds[6:8]],weights = [0.5, 0.5])
    evo_audioGM['lo40'][fp] = mne.combine_evoked([evo_audio[i][fp] for i in conds[8:10]],weights = [0.5, 0.5])
    evo_audioGM['lo80'][fp] = mne.combine_evoked([evo_audio[i][fp] for i in conds[10:12]],weights = [0.5, 0.5])
    evo_audioGM['hi20'][fp] = mne.combine_evoked([evo_audio[i][fp] for i in conds[12:14]],weights = [0.5, 0.5])
    evo_audioGM['hi40'][fp] = mne.combine_evoked([evo_audio[i][fp] for i in conds[14:16]],weights = [0.5, 0.5])
    evo_audioGM['hi80'][fp] = mne.combine_evoked([evo_audio[i][fp] for i in conds[16:18]],weights = [0.5, 0.5])
    # when I chose weights = 'equal', the weights were +1 each and the result was a sum of the ERPs


# plot mean amps and amplitude spectra for individual subjects
# ============================================================
#ylim = [-3, 3] # force a particular Y min and max
style = dict(colors={'No': 'black', 'Low': 'blue', 'High': 'red'},
    ci=None, show_sensors=False, legend='upper right',
    linestyles={'No': '-', 'Low': '-', 'High': '-'},
    truncate_xaxis=False, truncate_yaxis=False) #, ylim=dict(eeg=ylim))
plt.rcParams.update({'font.size': 14})

for fp in fps:
    # fp = 5
    #tmp_myfreqs = [myfreqs_20, myfreqs_40, myfreqs_80]
    tmp_names = [20,40,80]
                    # 20 no lo hi,  40 no lo hi,  80 no lo hi
    tmp_whichCon = [[0,2,6,8,12,14],[2,4,8,10,14,16],[4,6,10,12,16,18]]
    for f in range(3):
        # f = 0
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        pce({'No':  evo_audioGM['no{}'.format(tmp_names[f])][fp],
             'Low': evo_audioGM['lo{}'.format(tmp_names[f])][fp],
             'High': evo_audioGM['hi{}'.format(tmp_names[f])][fp]},
             **style, picks=chans, combine = 'mean',
             axes = ax[0])
        fpind = fps.index(fp)

        tmpno = np.mean(np.row_stack([Damp[c][fps.index(fp)] for c in conds[tmp_whichCon[f][0]:tmp_whichCon[f][1]]]), axis=0)
        tmplo = np.mean(np.row_stack([Damp[c][fps.index(fp)] for c in conds[tmp_whichCon[f][2]:tmp_whichCon[f][3]]]), axis=0)
        tmphi = np.mean(np.row_stack([Damp[c][fps.index(fp)] for c in conds[tmp_whichCon[f][4]:tmp_whichCon[f][5]]]), axis=0)

        ax[1].plot(myfreqs, tmpno, 'k-', label='No')
        ax[1].plot(myfreqs, tmplo, 'b-', label='Low')
        ax[1].plot(myfreqs, tmphi, 'r-', label='High')
        ax[1].legend(frameon=False)
        ax[1].set_xlabel('Frequencies (Hz)')
        ax[1].set_ylabel('Amplitudes (µV)')
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].set_ylim(bottom = 0.05, top = 0.5)
        ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[1].set_title("Amplitude spectrum")
        #tmp_myfreqs[f][indx_noise[7]]
        ax[1].axvspan(min(myfreqs[indx_noise]), myfreqs[indx_noise[7]], color='gray', alpha=0.2, lw=0)
        ax[1].axvspan(myfreqs[indx_noise[8]], max(myfreqs[indx_noise]), color='gray', alpha=0.2, lw=0)
        fig.suptitle("Subject "+str(fp))
        fig.tight_layout()
        plt.savefig(op.join(dir_results, 'figures', 'individual_aspctm', 'fp_{}_beta_for_{}Hz_AM.jpg'.format(fp, tmp_names[f])), format='jpg', dpi = 600)
        plt.savefig(op.join(dir_results, 'figures', 'individual_aspctm', 'fp_{}_beta_for_{}Hz_AM.pdf'.format(fp, tmp_names[f])), format='pdf', dpi = 600)
        plt.close(fig=None)

# plot grand mean amplitude spectrum by load
# ==========================================
for f in range(3):
    #f = 0
    #tmp_myfreqs = [myfreqs_20, myfreqs_40, myfreqs_80]
    tmp_names = [20,40,80]
    tmp_whichCon = [[0,2,6,8,12,14],[2,4,8,10,14,16],[4,6,10,12,16,18]]

    plt.close(fig=None)
    fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharey=False)
    style = dict(show_sensors=False, legend='upper right',
                 truncate_xaxis=False, truncate_yaxis=False, ci=None)
    pce({'No': ERP('no{}'.format(tmp_names[f]), evo_audioGM),
         'Low': ERP('lo{}'.format(tmp_names[f]), evo_audioGM),
         'High': ERP('hi{}'.format(tmp_names[f]), evo_audioGM)},
         colors={'No': 'black', 'Low': 'blue', 'High': 'red'},
         linestyles={'No': '-', 'Low': '-', 'High': '-'},
         **style, title=", ".join(chanpick), picks=chans, combine = 'mean',
         axes = ax[0])
        #ylim=dict(eeg=[-6, 12])
    ax[0].set_title('Mean ERP (top) and amplitude spectrum (bottom) (N = {})'.format(len(fps)))
    ax[0].set_ylabel('Amplitudes (µV)')
    ax[0].legend(frameon=False)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # for each load, combine the two blocks
    tmpno = np.mean(np.row_stack([np.row_stack([Damp[c][fps.index(fp)] for c in conds[tmp_whichCon[f][0]:tmp_whichCon[f][1]]]) for fp in fps]), axis=0)
    tmplo = np.mean(np.row_stack([np.row_stack([Damp[c][fps.index(fp)] for c in conds[tmp_whichCon[f][2]:tmp_whichCon[f][3]]]) for fp in fps]), axis=0)
    tmphi = np.mean(np.row_stack([np.row_stack([Damp[c][fps.index(fp)] for c in conds[tmp_whichCon[f][4]:tmp_whichCon[f][5]]]) for fp in fps]), axis=0)

    ax[1].plot(myfreqs, tmpno, 'k-', label='No')
    ax[1].plot(myfreqs, tmplo, 'b-', label='Low')
    ax[1].plot(myfreqs, tmphi, 'r-', label='High')
    ax[1].legend(frameon=False)
    ax[1].set_ylim(bottom = 0.05, top = 0.5)
    ax[1].set_xlabel('Frequencies (Hz)')
    ax[1].set_ylabel('Amplitudes (µV)')
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].axvspan(min(myfreqs[indx_noise]), myfreqs[indx_noise[7]], color='gray', alpha=0.2, lw=0)
    ax[1].axvspan(myfreqs[indx_noise[8]], max(myfreqs[indx_noise]), color='gray', alpha=0.2, lw=0)
    fig.tight_layout()
    fig.savefig(op.join(dir_results, 'figures', 'figure_average_aspctm_and_ERP_beta_for_{}Hz_N_{}.jpg'.format(tmp_names[f], len(fps))), format='jpg', dpi=600)
    fig.savefig(op.join(dir_results, 'figures', 'figure_average_aspctm_and_ERP_beta_for_{}Hz_N_{}.pdf'.format(tmp_names[f], len(fps))), format='pdf', dpi=600)
    plt.close(fig=None)


# plot grand mean itc by load
# ===========================
for f in range(3):
    #f = 0
    #tmp_myfreqs = [myfreqs_20, myfreqs_40, myfreqs_80]
    tmp_names = [20,40,80]
    tmp_whichCon = [[0,2,6,8,12,14],[2,4,8,10,14,16],[4,6,10,12,16,18]]

    plt.close(fig=None)
    fig, ax = plt.subplots(figsize=(14, 9), sharey=False) #subplot seems to be needed to define spines below
    # for each load, combine the four blocks
    tmpno = np.mean(np.row_stack([np.row_stack([Ditc[c][fps.index(fp)] for c in conds[tmp_whichCon[f][0]:tmp_whichCon[f][1]]]) for fp in fps]), axis=0)
    tmplo = np.mean(np.row_stack([np.row_stack([Ditc[c][fps.index(fp)] for c in conds[tmp_whichCon[f][2]:tmp_whichCon[f][3]]]) for fp in fps]), axis=0)
    tmphi = np.mean(np.row_stack([np.row_stack([Ditc[c][fps.index(fp)] for c in conds[tmp_whichCon[f][4]:tmp_whichCon[f][5]]]) for fp in fps]), axis=0)
    plt.plot(myfreqs, tmpno, 'k-', label='No')
    plt.plot(myfreqs, tmplo, 'b-', label='Low')
    plt.plot(myfreqs, tmphi, 'r-', label='High')
    plt.title('Intertrial coherence (itc)')
    plt.legend(frameon=False)
    plt.xlabel('Frequencies (Hz)')
    plt.ylabel('itc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom = 0.05, top = 0.2)
    ax.axvspan(min(myfreqs[indx_noise]), myfreqs[indx_noise[7]], color='gray', alpha=0.2, lw=0)
    ax.axvspan(myfreqs[indx_noise[8]], max(myfreqs[indx_noise]), color='gray', alpha=0.2, lw=0)
    fig.tight_layout()
    fig.savefig(op.join(dir_results, 'figures', 'figure_audio_itc_beta_for_{}_N_{}.jpg'.format(tmp_names[f],len(fps))), format='jpg', dpi=600)
    fig.savefig(op.join(dir_results, 'figures', 'figure_audio_itc_beta_for_{}_N_{}.pdf'.format(tmp_names[f],len(fps))), format='pdf', dpi=600)
    plt.close(fig=None)




