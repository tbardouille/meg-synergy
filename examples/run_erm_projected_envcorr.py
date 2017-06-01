import mne
import os.path as op
import matplotlib.pyplot as plt
from synergy.library.config import ctc, cal
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
import glob
import numpy as np
from scipy.signal import hilbert
from scipy.stats import ranksums
from synergy.utils import get_stc, make_surrogates_empty_room
import bct
import joblib


plt.ion()
subjects = ['CC110033', 'CC110037', 'CC110045']
subject = subjects[0]

# For Desktop
data_path = '/cluster/transcend/sheraz/Dropbox/mne-camcan-data/'

subjects_dir = op.join(data_path,'recons')
subject_dir = op.join(subjects_dir,subject)
bem_dir = op.join(subject_dir,'bem')
trans_file = op.join(data_path, 'trans',subject + '-trans.fif')
labels_fname  = glob.glob(op.join(data_path, 'labels', '*.label'))
labels = [mne.read_label(label, subject='fsaverageSK', color='r')
          for label in labels_fname]
for index, label in enumerate(labels):
    label.values.fill(1.0)
    labels[index] = label

labels = [label.morph('fsaverageSK', subject, subjects_dir=subjects_dir) for label in labels]

event_id = 1
event_overlap = 4
event_length = 30
spacing='ico5'


#def process_maxfilter(subject):
raw_fname = op.join(
    data_path, 'rest', 'sub-' + subject,'meg', 'rest_raw.fif')

erm_fname = op.join(
    data_path, 'emptyroom', '090430_raw.fif')


raw = mne.io.read_raw_fif(raw_fname, preload=True)
erm_raw = mne.io.read_raw_fif(erm_fname, preload=True)

raw_length = (raw.last_samp-raw.first_samp)/raw.info['sfreq']
raw.info['bads'] +=  [u'MEG2113', u'MEG1941', u'MEG1412', u'MEG2331']

erm_raw_length = (erm_raw.last_samp-erm_raw.first_samp)/erm_raw.info['sfreq']
erm_raw.info['bads'] +=  [u'MEG2113', u'MEG2112', u'MEG0512', u'MEG0141', u'MEG2533']


raw = mne.preprocessing.maxwell_filter(raw, calibration=cal,
                                       cross_talk=ctc,
                                       st_duration=35.,
                                       st_correlation=.96,
                                       origin=(0., 0., 0.04))

erm_raw = mne.preprocessing.maxwell_filter(erm_raw, calibration=cal,
                                       cross_talk=ctc,
                                       st_duration=35.,
                                       st_correlation=.96,
                                       coord_frame='meg',
                                       origin=(0., 0.013, -0.006))

projs_ecg, ecg_events = compute_proj_ecg(raw, n_grad=1, n_mag=2)
projs_eog, eog_events = compute_proj_eog(raw, n_grad=1, n_mag=2)

raw.info['projs'] += projs_ecg
raw.info['projs'] += projs_eog

erm_raw.info['projs'] += projs_ecg
erm_raw.info['projs'] += projs_eog

raw.apply_proj()
erm_raw.apply_proj()


raw.filter(None, 40, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero', fir_window='hann')

cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None)

erm_raw.filter(None, 40, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero', fir_window='hann')

erm_cov = mne.compute_raw_covariance(erm_raw, tmin=0, tmax=None)

raw.filter(8, 12, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero', fir_window='hann')
reject = dict(grad=1000e-13, mag=1.2e-12)
events = mne.make_fixed_length_events(raw, event_id, duration=event_overlap, start=0, stop=raw_length-event_length)
epochs = mne.Epochs(raw, events, event_id, 0,
                    event_length, baseline=None, preload=True, proj=False, reject=reject)
epochs.resample(100.)



bem_fname = op.join(bem_dir, '%s-src.fif' % subject)
src_fname = op.join(bem_dir, '%s-src.fif' % spacing)

bem = mne.read_bem_solution(bem_fname)
src = mne.read_source_spaces(src_fname)

fwd = mne.make_forward_solution(raw_fname, trans=trans_file, src=src, bem=bem, meg=True, eeg=False, n_jobs=2)
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, erm_cov,loose=0.2, depth=0.8)

snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
method = "MNE"
n_epochs, n_chs, n_time = epochs._data.shape

labels_data = np.zeros((len(labels), n_time, n_epochs))

for index, label in enumerate(labels):
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2, method, label, pick_ori="normal")
    data = np.transpose(np.array([stc.data for stc in stcs]), (1, 2, 0))
    n_verts, n_time, n_epochs = data.shape
    data = data.reshape(n_verts, n_time * n_epochs)
    U, S, V = np.linalg.svd(data, full_matrices=False)
    flip = np.array([np.sign(np.corrcoef(V[0,:],dat)[0, 1]) for dat in data])
    data = flip[:, np.newaxis] * data
    data = data.reshape(n_verts, n_time, n_epochs).mean(axis=0)
    labels_data[index] = data
    print(float(index) / len(labels) * 100)





fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True)
projected_erm_raw = make_surrogates_empty_room(erm_raw, fwd_fixed, inv, step=10000)


projected_erm_raw.filter(8, 12, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero', fir_window='hann')
projected_erm_events = mne.make_fixed_length_events(projected_erm_raw, event_id,
                                                    duration=event_overlap, start=0,
                                                    stop=erm_raw_length-event_length)
projected_erm_epochs = mne.Epochs(projected_erm_raw, projected_erm_events, event_id, 0,
                    event_length, baseline=None, preload=True, proj=False, reject=reject)
projected_erm_epochs.resample(100.)

n_epochs, n_chs, n_time = projected_erm_epochs._data.shape

labels_data_erm = np.zeros((len(labels), n_time, n_epochs))

for index, label in enumerate(labels):
    stcs = mne.minimum_norm.apply_inverse_epochs(projected_erm_epochs, inv, lambda2, method, label, pick_ori="normal")
    data = np.transpose(np.array([stc.data for stc in stcs]), (1, 2, 0))
    n_verts, n_time, n_epochs = data.shape
    data = data.reshape(n_verts, n_time * n_epochs)
    U, S, V = np.linalg.svd(data, full_matrices=False)
    flip = np.array([np.sign(np.corrcoef(V[0,:],dat)[0, 1]) for dat in data])
    data = flip[:, np.newaxis] * data
    data = data.reshape(n_verts, n_time, n_epochs).mean(axis=0)
    labels_data_erm[index] = data
    print(float(index) / len(labels) * 100)




labels_data_hipp = labels_data.copy()
labels_data_hipp = hilbert(labels_data_hipp, axis=1)


corr_mats = np.zeros((len(labels),len(labels), n_epochs))

for index, label_data in enumerate(labels_data_hipp):
    label_data_orth = np.imag(label_data*(labels_data_hipp.conj()/np.abs(labels_data_hipp)))
    label_data_orig = np.abs(label_data)
    label_data_cont = np.transpose(np.dstack((label_data_orig,
                                          np.transpose(label_data_orth, (1, 2, 0)))), (1 ,2, 0))
    corr_mats[index] = np.array([np.corrcoef(dat) for dat in label_data_cont])[:,0,1:].T
    print(float(index)/len(labels)*100)

corr_mats = np.transpose(corr_mats,(2,0,1))

corr_hipp = np.median(np.array([(np.abs(corr_mat) + np.abs(corr_mat).T)/2.
                        for corr_mat in corr_mats]),axis=0)




labels_data = np.abs(hilbert(labels_data, axis=1))
labels_data_erm = np.abs(hilbert(labels_data_erm, axis=1))

labels_data = np.transpose(labels_data,(2,0,1))
labels_data_erm = np.transpose(labels_data_erm,(2,0,1))

corr_rest = np.abs([np.corrcoef(np.log10(dat)) for dat in labels_data])
corr_erm = np.abs([np.corrcoef(np.log10(dat)) for dat in labels_data_erm])

corr_z =  np.zeros((len(labels), len(labels)))
for index1 in range(len(labels)):
    for index2 in range(len(labels)):
        corr_z[index1, index2] = ranksums(np.log10(corr_rest[:, index1, index2]),np.log10(corr_erm[:, index1, index2]))[0]
        print((index1, index2))





corr_rest_median = np.median(corr_rest,0)
corr_erm_median = np.median(corr_erm,0)


ll = [label_fname.split('/')[-1] for label_fname in labels_fname]

temporal_labels_indices = [index  for index,l in enumerate(ll) if 'lh' in l and 'sup' in l and 'marginal' in l]




data = {'corr_rest_median': corr_rest_median, 'corr_erm_median':corr_erm_median,
        'corr_z':corr_z, 'corr_hipp':corr_hipp, 'labels_fname':labels_fname}

hkl_fname = os.path.join(dir_path,'corrs.hkl')
hkl.dump(data, hkl_fname)

##
projected_erm_cov = mne.compute_raw_covariance(projected_erm_raw, tmin=0, tmax=None)
cov.plot(raw.info)
erm_cov.plot(erm_raw.info)
projected_erm_cov.plot(projected_erm_raw.info)

corr_zz = corr_z.copy()

corr_zz[corr_zz<0] = 0

stc = get_stc(labels_fname, corr_z[temporal_labels_indices,:].mean(0))
brain = stc.plot(subject='fsaverageSK', time_viewer=True,hemi='split', colormap='gnuplot',
                           views=['lateral','medial'],
                 surface='inflated10', subjects_dir=subjects_dir,clim={'kind':'value','lims':[5,7.5,10]})

stc = get_stc(labels_fname, corr_rest_median[temporal_labels_indices,:].mean(0))
brain = stc.plot(subject='fsaverageSK', time_viewer=True,hemi='split', colormap='gnuplot',
                           views=['lateral','medial'],
                 surface='inflated10', subjects_dir=subjects_dir,clim={'kind':'value','lims':[0,.5,1]})


corr = np.int32(bct.utils.threshold_proportional(corr_z,.15) > 0)
deg = bct.degrees_und(corr)

stc = get_stc(labels_fname, corr_z[254,:])
brain = stc.plot(subject='fsaverageSK', time_viewer=True,hemi='split', colormap='gnuplot',
                           views=['lateral','medial'],
                 surface='inflated10', subjects_dir=subjects_dir,clim={'kind':'value','lims':[5,6,7]})

brain.save_image('beta_projected_erm_corr.png')