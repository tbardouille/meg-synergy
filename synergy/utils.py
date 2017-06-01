# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 15:12:09 2016
@author: sheraz
"""
import mne
from mne.minimum_norm.inverse import (read_inverse_operator, combine_xyz, prepare_inverse_operator, _assemble_kernel)

from mne import convert_forward_solution, read_forward_solution
from mne.io.constants import FIFF
from mne.io import read_raw_fif
from mne import EpochsArray
import numpy as np
from mne.utils import logger
from mne.time_frequency.tfr import _check_decim, morlet, cwt
from mne.parallel import parallel_func
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import fnmatch
import h5py
import os, errno
import csv, re
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
import os.path as op
import subprocess
import time
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
import copy


class Flexlist(list):
    def __getitem__(self, keys):
        if isinstance(keys, (int, slice)): return list.__getitem__(self, keys)
        return [self[k] for k in keys]


def mad(data):
    return np.median(np.absolute(data - np.median(data)))


def computeZscore(data):
    return (data - np.mean(data)) / np.std(data)


def computeNPZscore(data):
    return (data - np.median(data)) / mad(data)


def getInversionKernel(fname_inv, nave=1, lambda2=1. / 9., method='MNE', label=None, pick_ori=None):
    inverse_operator = read_inverse_operator(fname_inv)
    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, method)
    K, noise_norm, vertno = _assemble_kernel(inv, label, 'MNE', pick_ori)
    is_free_ori = inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI
    return K, noise_norm, vertno, is_free_ori


def applyInverse(data, K, noise_norm, vertno, is_free_ori):
    sol = np.dot(K, data)
    if is_free_ori:
        sol = combine_xyz(sol)
    if noise_norm:
        sol *= noise_norm
    return sol, K, noise_norm, vertno


def findIndexGradMag(raw):
    indexGrad = np.array(
        [index for index, item in enumerate(raw.ch_names) if (item[-1] == '2' or item[-1] == '3') and item[0] == 'M'])
    indexMag = np.array([index for index, item in enumerate(raw.ch_names) if (item[-1] == '1') and item[0] == 'M'])
    return indexGrad, indexMag


def getGainMatrix(fname_fwd, surf_ori=False, force_fixed=True):
    fwd = convert_forward_solution(read_forward_solution(fname_fwd),
                                   surf_ori, force_fixed)
    F = fwd['sol']['data']
    return F


def applyForward(data, F):
    B = np.dot(F, data)
    return B


def createEpochObject(data, info, events=None, tmin=0, event_id=None):
    if type(info) == str:
        info = mne.io.read_info(info)
    if not events:
        events = np.array((np.arange(data.shape[0]), np.ones(data.shape[0]), np.ones(data.shape[0]))).T
    if not event_id:
        event_id = event_id = {'arbitrary': 1}
    epochs = EpochsArray(data, info=info, events=events, event_id=event_id)
    return epochs


def readRaw(fname_raw, chansel=np.arange(306), startTime=0, endTime=-1):
    raw = read_raw_fif(fname_raw)
    startTime = raw.time_as_index(startTime)
    if endTime == -1:
        data, times = raw[chansel, startTime:]
    else:
        endTime = raw.time_as_index(endTime)
        data, times = raw[chansel, startTime:(endTime + 1)]
    return data, times, raw


def createCustomEvents():
    pass


def overlapTriggersResting(raw, eventDistance=8, event_id=1):
    sampRate = raw.info['sfreq']
    start_samp = raw.first_samp
    end_samp = raw.last_samp
    totalLength = end_samp - start_samp + 1 - np.ceil(sampRate * eventDistance)
    eventsSamples = np.arange(0., totalLength, np.ceil(sampRate * eventDistance))
    events = np.empty((eventsSamples.shape[0], 3), dtype=int)
    for ind, samp in enumerate(eventsSamples):
        events[ind, :] = start_samp + samp, 0, event_id
    return events


def detectBadChannels(raw, zscore=3):
    raw_copy = raw.copy()
    indexGrad, indexMag = findIndexGradMag(raw_copy)
    raw_copy = raw_copy.crop(30., 120.).load_data().filter(0.5, 50).resample(150, npad='auto')
    # raw_copy = raw_copy.crop(30., 90.).load_data()
    sfreq = int(raw_copy.info['sfreq'])
    dataG = np.sqrt(np.sum(raw_copy[indexGrad, sfreq * 10:-sfreq * 10][0] ** 2, axis=1))
    dataM = np.sqrt(np.sum(raw_copy[indexMag, sfreq * 10:-sfreq * 10][0] ** 2, axis=1))
    max_th_G = computeZscore(dataG)
    max_th_M = computeZscore(dataM)
    indexGrad_B = np.array(indexGrad)[np.abs(max_th_G) > zscore]
    indexMag_B = np.array(indexMag)[np.abs(max_th_M) > zscore]
    return Flexlist(raw.info['ch_names'])[indexGrad_B] + Flexlist(raw.info['ch_names'])[indexMag_B]


def doMaxFilter(fname_raw):
    pass


def doFilter(fname_raw):
    pass


def compute_ecg_eog_proj(raw, ecg=True, eog=True, n_grad_ecg=1, n_grad_eog=1,
                         n_mag_ecg=1, n_mag_eog=1, average=True, vis=False,
                         apply=True):
    if ecg:
        projs, events_ecg = compute_proj_ecg(raw, n_grad=n_grad_ecg,
                                             n_mag=n_mag_ecg, average=average)
        ecg_projs = projs[-(n_mag_ecg + n_grad_ecg):]
        if apply:
            raw.info['projs'] += ecg_projs
        if vis:
            layout = mne.channels.read_layout('Vectorview-all')
            names = [name.replace(' ', '') for name in layout.names]
            layout.names = names
            mne.viz.plot_projs_topomap(ecg_projs, layout=layout)

    # Now for EOG
    if eog:
        projs, events_eog = compute_proj_eog(raw, n_grad=n_grad_eog, n_mag=n_mag_eog,
                                             average=average)
        eog_projs = projs[-(n_mag_eog + n_grad_eog):]
        if apply:
            raw.info['projs'] += eog_projs
        if vis:
            layout = mne.channels.read_layout('Vectorview-all')
            names = [name.replace(' ', '') for name in layout.names]
            layout.names = names
            mne.viz.plot_projs_topomap(eog_projs, layout=layout)

    return ecg_projs, eog_projs, events_ecg, events_eog


def readMatFile(fname_mat, variable_needed='all'):
    f = h5py.File(fname_mat, 'r')
    variables = f.items()
    matData = dict()
    for var in variables:
        name = var[0]
        data = var[1]
        if variable_needed != 'all' and name not in variable_needed:
            continue
        print "Variable Name ", name  # Name
        if type(data) is h5py.Dataset:
            matData.update({name: data.value})
    return matData


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occured


def find_badchannels_csv(filename, subject_id):
    with open(filename, 'rt') as f:
        reader = csv.reader(f)
        next(reader, None)
        badchannels = [row[-2] for row in reader if row[0] == subject_id]
        if badchannels:
            badchannels = badchannels[0].split(' ')
            badchannels = [badchannel for badchannel in badchannels if badchannel != '' and badchannel != ' ']
        return badchannels


def run_ica(raw, ica_fname=None, picks=None, n_max_ecg=3, n_max_eog=1):
    raw_f = raw.copy().filter(1, 45, n_jobs=1, l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                              filter_length='10s', phase='zero-double')

    ica = ICA(method='fastica', random_state=42, n_components=0.98)
    if picks is None:
        picks = mne.pick_types(raw_f.info, meg=True, eeg=False, eog=False,
                               stim=False, exclude='bads')
    ica.fit(raw_f, picks=picks, reject=dict(grad=4000e-13, mag=4e-12),
            decim=3)
    ecg_epochs = create_ecg_epochs(raw_f, tmin=-.5, tmax=.5, picks=picks)
    ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
    ecg_inds = ecg_inds[:n_max_ecg]
    ica.exclude += ecg_inds
    eog_inds, scores = ica.find_bads_eog(raw_f)
    eog_inds = eog_inds[:n_max_eog]
    ica.exclude += eog_inds
    if ica_fname is None:
        ica_fname = raw_f._filenames[0][:-4] + '-pyimpress-ica.fif'
    ica.save(ica_fname)
    return ica, ica_fname


def run_events(raw, events_fname=None):
    mask = 4096 + 256  # mask for excluding high order bits
    events = mne.find_events(raw, stim_channel='STI101',
                             consecutive='increasing', mask=mask,
                             mask_type='not_and', min_duration=0.0001,
                             verbose=True)

    if events_fname is None:
        events_fname = raw._filenames[0][:-4] + '-pyimpress-eve.fif'
    mne.write_events(events_fname, events)
    return events, events_fname


def run_epochs(raw, events, picks=None, tmin=-0.2, tmax=0.6,
               reject=dict(grad=4000e-13, mag=4e-12, eog=350e-6),
               baseline=(-0.2, 0), proj=True,
               event_id=dict(event_1=1, event_2=2, event_3=3, event_4=4)):
    if picks is None:
        picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True, stim=False,
                               exclude='bads')

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=baseline, reject=reject, proj=proj)
    return epochs


def run_mri(subject, type='td', visit=1, run=1, email='sheraz@nmr.mgh.harvard.edu'):
    COMMAND = ['/eris/p41p3/transcend/scripts/MRI/copymri.sh', '-s', '%s' % subject,
               '-t', '%s' % type, '-v', '%s' % visit, '-r', '%s' % run, '-e', '%s' % email]

    p = subprocess.Popen(COMMAND, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()


def run_strural(subject, bem_ico=4, spacing='ico5', n_jobs=4,
                subjects_dir='/cluster/transcend/MRI/WMA/recons'):
    mne.bem.make_watershed_bem(subject, subjects_dir=subjects_dir,
                               overwrite=True)
    src_fname = op.join(subjects_dir, subject, '%s-pyimpress-src.fif' % spacing)
    if not os.path.isfile(src_fname):

        src = mne.setup_source_space(subject, spacing=spacing,
                                     subjects_dir=subjects_dir, overwrite=True,
                                     n_jobs=n_jobs, add_dist=True)
        mne.write_source_spaces(src_fname, src)
    else:
        src = mne.read_source_spaces(src_fname)

    bem_fname = op.join(subjects_dir, subject, '%s-pyimpress-bem.fif' % bem_ico)

    if not os.path.isfile(bem_fname):
        bem_model = mne.make_bem_model(subject, ico=bem_ico, subjects_dir=subjects_dir,
                                       conductivity=(0.3,))
        bem = mne.make_bem_solution(bem_model)
        mne.write_bem_solution(bem_fname, bem)
    else:
        bem = mne.read_bem_solution(bem_fname)
    return src, bem, src_fname, bem_fname


def run_fwd_inv(fname_raw, subject, cov, fname_trans, subjects_dir='/cluster/transcend/MRI/WMA/recons',
                src=None, bem=None, fname_fwd=None, meg=True, fname_inv=None,
                eeg=False, n_jobs=2, bem_ico=4,
                spacing='ico5', loose=0.2, rank=None,
                depth=0.8, fixed=True, limit_depth_chs=True, reject=None):
    if src is None:
        src_fname = op.join(subjects_dir, subject, '%s-pyimpress-src.fif' % spacing)
        if not os.path.isfile(src_fname):
            src = run_strural(subject, bem_ico=bem_ico, spacing=spacing, subjects_dir=subjects_dir)[0]
        else:
            src = mne.read_source_spaces(src_fname)

    if bem is None:
        bem_fname = op.join(subjects_dir, subject, '%s-pyimpress-bem.fif' % bem_ico)
        if not os.path.isfile(bem_fname):
            bem = run_strural(subject, bem_ico=bem_ico, spacing=spacing, subjects_dir=subjects_dir)[1]
        else:
            bem = mne.read_bem_solution(bem_fname)

    if fname_fwd is None:
        fname_fwd = fname_raw[:-4] + '-fwd.fif'
        if not os.path.isfile(fname_fwd):

            if os.path.isfile(fname_trans) and os.path.isfile(fname_raw):
                raw = mne.io.read_raw_fif(fname_raw)
                fwd = mne.make_forward_solution(fname_raw, trans=fname_trans, src=src, bem=bem,
                                                fname=fname_fwd, meg=meg, eeg=eeg, n_jobs=n_jobs)
                mne.write_forward_solution(fname_fwd, fwd, overwrite=True)
            else:
                raise Exception('fname_trans and fname_raw both needed')

        else:
            fwd = mne.read_forward_solution(fname_fwd)
            if os.path.isfile(fname_raw):
                raw = mne.io.read_raw_fif(fname_raw)

    if fname_inv is None:
        if fixed:
            fname_inv = fname_raw[:-4] + '-fixed-inv.fif'
        else:
            fname_inv = fname_raw[:-4] + '-loose-inv.fif'
        if not os.path.isfile(fname_inv):
            if cov is not None:
                inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov,
                                                             loose=loose, depth=depth,
                                                             fixed=False, limit_depth_chs=limit_depth_chs, rank=rank)
                mne.minimum_norm.write_inverse_operator(fname_inv, inv)
        else:
            inv = mne.minimum_norm.read_inverse_operator(fname_inv)

    return fwd, inv


def compute_cov(cov_object, reject=None, tmax=0, method='shrunk'):
    if isinstance(cov_object, mne.epochs._BaseEpochs):
        cov = mne.compute_covariance(cov_object, tmax=tmax, method=method)
    elif isinstance(cov_object, mne.io.base._BaseRaw):
        if reject is None:
            reject, picks = compute_reject_picks(cov_object)
        cov = mne.compute_raw_covariance(cov_object, reject=reject, picks=picks)
    return cov


def compute_reject_picks(raw):
    if any(ch.startswith('EOG') for ch in raw.info['ch_names']) \
            and any(ch.startswith('EEG') for ch in raw.info['ch_names']):
        reject = dict(grad=4000e-13, mag=4e-12, eog=250e-6, eeg=180e-6)
        picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True, stim=False)
    elif any(ch.startswith('EOG') for ch in raw.info['ch_names']):
        reject = dict(grad=4000e-13, mag=4e-12, eog=250e-6)
        picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True, stim=False)
    else:
        reject = dict(grad=4000e-13, mag=4e-12)
        picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False)
    return reject, picks


def mat2mne(data, chan_names='meg', chan_types=None, sfreq=250, events=None,
            tmin=0):
    from mne.epochs import EpochsArray
    from mne.io.meas_info import create_info
    data = np.array(data)
    print('Trials: {}, Labels: {}, TimePoints: {}'.format(*data.shape))
    n_trial, n_chan, n_time = data.shape
    # chan names
    if isinstance(chan_names, str):
        chan_names = [chan_names + '_%02i' % chan for chan in range(n_chan)]
    if len(chan_names) != n_chan:
        raise ValueError('chan_names must be a string or a list of'
                         'n_chan strings')
    # chan types
    if isinstance(chan_types, str):
        chan_types = [chan_types] * n_chan
    elif chan_types is None:
        if isinstance(chan_names, str):
            if chan_names != 'meg':
                chan_types = [chan_names] * n_chan
            else:
                chan_types = ['mag'] * n_chan
        elif isinstance(chan_names, list):
            chan_types = ['mag' for chan in chan_names]
        else:
            raise ValueError('Specify chan_types')

    # events
    if events is None:
        events = np.c_[np.cumsum(np.ones(n_trial)) * 5 * sfreq,
                       np.zeros(n_trial, int), np.zeros(n_trial)]
    else:
        events = np.array(events, int)
        if events.ndim == 1:
            events = np.c_[np.cumsum(np.ones(n_trial)) * 5 * sfreq,
                           np.zeros(n_trial), events]
        elif (events.ndim != 2) or (events.shape[1] != 3):
            raise ValueError('events shape must be ntrial, or ntrials * 3')

    info = create_info(chan_names, sfreq, chan_types)
    return EpochsArray(data, info, events=np.array(events, int), verbose=False,
                       tmin=tmin)


def list2csv(subjs, fname_csv):
    import csv
    with open(fname_csv, "wb") as f:
        writer = csv.writer(f, lineterminator='\n')
        for subj in subjs:
            writer.writerow([subj])

    return fname_csv


def csv2dict(csv_fname):
    reader = csv.DictReader(open(csv_fname))
    result = {}
    for row in reader:
        for column, value in row.iteritems():
            result.setdefault(column, []).append(value)
    return result


def labels2stc(labels,labels_data,stc):
    stc_new = stc.copy()
    stc_new.data.fill(0)
    for index,label in enumerate(labels):
        if labels_data.ndim==1:
            temp = stc.in_label(mne.read_label(label))
            temp.data.fill(labels_data[index])
            stc_new += temp.expand(stc.vertices)
        else:
            lab = mne.read_label(label)
            ver = np.intersect1d(lab.vertices, stc.vertices)
            if '-rh' in label:
                ver=ver+len(stc.vertices[0])
            stc_data = np.tile(labels_data[index,:][:,np.newaxis],len(ver)).T
            stc_new.data[ver, :] = stc_data
    return stc_new


def get_stc(labels,data,tmin=0,tstep=1):
    stc_vertices = [np.uint32(np.arange(10242)), np.uint32(np.arange(10242))]

    if data.ndim==1:
        stc_data = np.ones((20484,1), dtype=np.float32)
    else:
        stc_data = np.ones((20484, data.shape[1]), dtype=np.float32)

    stc = mne.SourceEstimate(stc_data, vertices=stc_vertices, tmin=tmin, tstep=tstep, subject='fsaverage')
    stc_new = labels2stc(labels, data, stc)
    return stc_new


def time_usage(func):
    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        func(*args, **kwargs)
        end_ts = time.time()
        print("elapsed time: %f" % (end_ts - beg_ts))

    return wrapper


def lin_reg(x, y, pvalue=0.05):
    # fit a curve to the data using a least squares 1st order polynomial fit
    import scipy.stats.t as tstats
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    fit = p(x)

    # get the coordinates for the fit curve
    c_y = [np.min(fit), np.max(fit)]
    c_x = [np.min(x), np.max(x)]

    # predict y values of origional data using the fit
    p_y = z[0] * x + z[1]

    # calculate the y-error (residuals)
    y_err = y - p_y

    # create series of new test x-values to predict for
    p_x = np.arange(np.min(x), np.max(x) + 1, 1)

    # now calculate confidence intervals for new test x-series
    mean_x = np.mean(x)  # mean of x
    n = len(x)  # number of samples in origional fit
    t = tstats.ppf(1 - pvalue, len(x))  # appropriate t value (where n=9, two tailed 95%)
    s_err = np.sum(np.power(y_err, 2))  # sum of the squares of the residuals

    confs = t * np.sqrt((s_err / (n - 2)) * (1.0 / n + (np.power((p_x - mean_x), 2) /
                                                        ((np.sum(np.power(x, 2))) - n * (np.power(mean_x, 2))))))

    # now predict y based on test x-values
    p_y = z[0] * p_x + z[0]

    # get lower and upper confidence limits based on predicted y and confidence intervals
    lower = p_y - abs(confs)
    upper = p_y + abs(confs)
    return c_x, c_y, p_x, lower, upper


def print_test():
    print('I am done')

def make_surrogates_empty_room(raw, fwd, inverse_operator, step=10000):
    """Create spatially structured noise from empty room MEG

    .. note::
        Convert MEG empty room to spatially structured noise by applying
        the inverse solution and then the forward solution.

    .. note::
        To safe memory, projection proceeds in non-overlapping sliding windows.

    Parameters
    ----------
    raw : instance of mne.io.Raw
        The raw data (empty room).
    fwd : instance of mne.Forward
        The forward solution
    inverse_operator : mne.minimum_norm.InverseOperator
        The inverse solution.
    step : int
        The step size (in samples) when iterating over the raw object.

    Returns
    -------
    raw_surr : instance of mne.io.Raw
        The surrogate MEG data.
    """
    index = np.arange(len(raw.times)).astype(int)
    out = np.empty(raw.get_data().shape, dtype=raw.get_data().dtype)
    picks = mne.pick_types(raw.info, meg=True, eeg=True, ref_meg=False)
    other_picks = [ii for ii in range(len(raw.ch_names)) if ii not in picks]
    out[other_picks] = raw.get_data()[other_picks]
    last = len(index)
    for start in index[::step]:
        stop = start + min(step, last - start)
        stc = mne.minimum_norm.apply_inverse_raw(
            raw, inverse_operator, lambda2=1.0, method='MNE', start=start,
            stop=stop, pick_ori="normal")
        reprojected = mne.apply_forward(fwd=fwd, stc=stc, info=raw.info)
        out[picks, start:stop] = reprojected.data
    out = mne.io.RawArray(out, info=copy.deepcopy(raw.info))
    return out

