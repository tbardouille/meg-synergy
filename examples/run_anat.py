import os.path as op
from nipype.interfaces.freesurfer import ReconAll
from mne.parallel import parallel_func
import glob

subjects_dir = '/cluster/fusion/Sheraz/camcan/freesurfer'
camcan_path = '/cluster/transcend/MEG'


N_JOBS = 240
t1_files = op.join(camcan_path + '/camcan47/cc700/mri/pipeline/release004/BIDSsep/anat/sub-' + '*',
             'anat', 'sub-' + '*' + '_T1w.nii.gz')


t1_files = glob.glob(t1_files)



def process_subject_anatomy(t1):
    reconall = ReconAll()
    reconall.inputs.subject_id = t1.split('/')[-3]
    reconall.inputs.directive = 'all'
    reconall.inputs.subjects_dir = subjects_dir
    reconall.inputs.T1_files = t1
    reconall.run()


parallel, run_func, _ = parallel_func(process_subject_anatomy, n_jobs=N_JOBS)
parallel(run_func(t1) for t1 in t1_files)



