import mne
import os.path as op
import matplotlib.pyplot as plt
from synergy.library.config import ctc, cal
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
import glob
import numpy as np
from scipy.signal import hilbert
from synergy.utils import get_stc
import bct

subjects = ['CC110033', 'CC110037', 'CC110045']
subject = subjects[0]

data_path = '/home/sheraz/Dropbox/mne-camcan-data'
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
event_overlap = 8
event_length = 30
spacing='ico5'


#def process_maxfilter(subject):
raw_fname = op.join(
    data_path, 'rest', 'sub-' + subject,'meg', 'rest_raw.fif')
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw_length = (raw.last_samp-raw.first_samp)/raw.info['sfreq']



raw.info['bads'] +=  [u'MEG2113', u'MEG1941', u'MEG1412', u'MEG2331']


raw = mne.preprocessing.maxwell_filter(raw, calibration=cal,
                                       cross_talk=ctc,
                                       st_duration=35.,
                                       st_correlation=.96,
                                       origin=(0., 0., 0.04))

projs_ecg, ecg_events = compute_proj_ecg(raw, n_grad=1, n_mag=2)
projs_eog, eog_events = compute_proj_eog(raw, n_grad=1, n_mag=2)

raw.info['projs'] += projs_ecg
raw.info['projs'] += projs_eog

raw.apply_proj()

raw.filter(None, 40, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero', fir_window='hann')

cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None)

raw.filter(14, 30, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
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
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov,loose=0.2, depth=0.8)

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
    data = np.median(data.reshape(n_verts, n_time, n_epochs), axis=0)
    labels_data[index] = data
    print(float(index) / len(labels) * 100)

labels_data = hilbert(labels_data, axis=1)
corr_mats = np.zeros((len(labels),len(labels), n_epochs))

for index, label_data in enumerate(labels_data):
    label_data_orth = np.imag(label_data*(labels_data.conj()/np.abs(labels_data)))
    label_data_orig = np.abs(label_data)
    label_data_cont = np.transpose(np.dstack((label_data_orig,
                                          np.transpose(label_data_orth, (1, 2, 0)))), (1 ,2, 0))
    corr_mats[index] = np.array([np.corrcoef(dat) for dat in label_data_cont])[:,0,1:].T
    print(float(index)/len(labels)*100)

corr_mats = np.transpose(corr_mats,(2,0,1))

corr = np.median(np.array([(np.abs(corr_mat) + np.abs(corr_mat).T)/2.
                        for corr_mat in corr_mats]),axis=0)

corr = np.int32(bct.utils.threshold_proportional(corr,.15) > 0)
deg = bct.density_und(corr)

stc = get_stc(labels_fname, deg)
brain = stc.plot(subject='fsaverageSK', time_viewer=True,hemi='split', colormap='gnuplot',
                           views=['lateral','medial'],
                 surface='inflated10', subjects_dir=subjects_dir)

brain.save_image('beta_orthogonal_corr.png')
