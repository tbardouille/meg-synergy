import mne
from mne.parallel import parallel_func
import os.path as op
import glob

subjects_dir = '/cluster/fusion/Sheraz/camcan/recons'
camcan_path = '/cluster/transcend/MEG'

mne.set_config('SUBJECTS_DIR',subjects_dir)


N_JOBS = 256

t1_files = op.join(camcan_path + '/camcan47/cc700/mri/pipeline/release004/BIDSsep/anat/sub-' + '*',
             'anat', 'sub-' + '*' + '_T1w.nii.gz')


t1_files = glob.glob(t1_files)

subjects = [t1.split('/')[-3] for t1 in t1_files]


def process_subject_bem(subject, spacing='ico5'):
    mne.bem.make_watershed_bem(subject=subject, subjects_dir=subjects_dir, overwrite=True, volume='T1', atlas=True,
                       gcaatlas=False, preflood=None)
    conductivity = (0.3,)
    model = mne.make_bem_model(subject=subject, ico=4,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    src = mne.setup_source_space(subject, spacing=spacing,
                                 subjects_dir=subjects_dir,
                                 add_dist=False, overwrite=True)
    bem_fname = op.join(subjects_dir,subject,'bem', '%s-src.fif' % subject)
    src_fname = op.join(subjects_dir, subject, 'bem', '%s-src.fif' % spacing)
    mne.write_bem_solution(bem_fname, bem=bem)
    mne.write_source_spaces(src_fname, src=src, overwrite=True)


parallel, run_func, _ = parallel_func(process_subject_bem, n_jobs=N_JOBS)
parallel(run_func(subject) for subject in subjects)

