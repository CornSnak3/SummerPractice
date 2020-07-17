import argparse
import logging
from s4d.utils import *
from sidekit import Mixture, FeaturesServer, StatServer
from sidekit.frontend.io import write_spro4
from sidekit.frontend.vad import vad_energy
from sidekit.sidekit_io import write_matrix, write_vect, init_logging
from s4d.diar import Diar

try:
    from sortedcontainers import SortedDict as dict
except ImportError:
    pass

init_logging(level=logging.INFO)
num_thread = 4
audio_dir = '../data/train/{}.wav'


ubm_seg_fn = './data/seg/ubm_ester.seg'
nb_gauss = 1024
mfcc_ubm_fn = './data/mfcc/ubm.h5'
ubm_idmap_fn = './data/mfcc/ubm_idmap.txt'
ubm_fn = './data/model/ester_ubm_'+str(nb_gauss)+'.h5'


tv_seg_fn = './data/seg/train.tv.seg'
rank_tv = 300
it_max_tv = 10
mfcc_tv_fn = './data/mfcc/tv.h5'
tv_idmap_fn = './data/mfcc/tv_idmap.h5'
tv_stat_fn  = './data/model/tv.stat.h5'
tv_fn = './data/model/tv_'+str(rank_tv)+'.h5'


plda_seg_fn = './data/seg/train.plda.seg'
rank_plda = 150
it_max_plda = 10
mfcc_plda_fn = './data/mfcc/norm_plda.h5'
plda_idmap_fn = './data/mfcc/plda_idmap.h5'
plda_fn = './data/model/plda_'+str(rank_tv)+'_'+str(rank_plda)+'.h5'
norm_stat_fn = './data/model/norm.stat.h5'
norm_fn = './data/model/norm.h5'
norm_iv_fn = './data/model/norm.iv.h5'


matrices_fn = './data/model/matrices.h5'
model_fn = './data/model/ester_model_{}_{}_{}.h5'.format(nb_gauss, rank_tv, rank_plda)

logging.info('Computing MFCC for UBM')
diar_ubm = Diar.read_seg(ubm_seg_fn, normalize_cluster=True)
fe = get_feature_extractor(audio_dir, 'sid')
ubm_idmap = fe.save_multispeakers(diar_ubm.id_map(), output_feature_filename=mfcc_ubm_fn, keep_all=False)
ubm_idmap.write_txt(ubm_idmap_fn)

fs = get_feature_server(mfcc_ubm_fn, 'sid')

spk_lst = ubm_idmap.rightids
ubm = Mixture()
ubm.EM_split(fs, spk_lst, nb_gauss,
             iterations=(1, 2, 2, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8), num_thread=num_thread,
             llk_gain=0.01)
ubm.write(ubm_fn, prefix='ubm/')

logging.info('Computing MFCC for TV')
diar_tv = Diar.read_seg(tv_seg_fn, normalize_cluster=True)
fe = get_feature_extractor(audio_dir, 'sid')
tv_idmap = fe.save_multispeakers(diar_tv.id_map(), output_feature_filename=mfcc_tv_fn, keep_all=False)
tv_idmap.write(tv_idmap_fn)

tv_idmap = IdMap.read(tv_idmap_fn)

ubm = Mixture()
ubm.read(ubm_fn, prefix='ubm/')

fs = get_feature_server(mfcc_tv_fn, 'sid')

tv_idmap.leftids = numpy.copy(tv_idmap.rightids)

tv_stat = StatServer(tv_idmap, ubm.get_distrib_nb(), ubm.dim())
tv_stat.accumulate_stat(ubm=ubm, feature_server=fs, seg_indices=range(tv_stat.segset.shape[0]), num_thread=num_thread)
tv_stat.write(tv_stat_fn)
fa = FactorAnalyser()
fa.total_variability(tv_stat_fn, ubm, rank_tv, nb_iter=it_max_tv, batch_size=1000, num_thread=num_thread)

write_tv_hdf5([fa.F, fa.mean, fa.Sigma], tv_fn)

logging.info('Computing MFCC for PLDA')
diar_plda = Diar.read_seg(plda_seg_fn, normalize_cluster=True)
fe = get_feature_extractor(audio_dir, 'sid')
plda_idmap = fe.save_multispeakers(diar_plda.id_map(), output_feature_filename=mfcc_plda_fn, keep_all=False)
plda_idmap.write(plda_idmap_fn)

plda_idmap = IdMap.read(plda_idmap_fn)

ubm = Mixture()
ubm.read(ubm_fn, prefix='ubm/')
tv, tv_mean, tv_sigma = read_tv_hdf5(tv_fn)

fs = get_feature_server(mfcc_plda_fn, 'sid')

plda_norm_stat = StatServer(plda_idmap, ubm.get_distrib_nb(), ubm.dim())
plda_norm_stat.accumulate_stat(ubm=ubm, feature_server=fs,
                               seg_indices=range(plda_norm_stat.segset.shape[0]), num_thread=num_thread)
plda_norm_stat.write(norm_stat_fn)

fa = FactorAnalyser(F=tv, mean=tv_mean, Sigma=tv_sigma)
norm_iv = fa.extract_ivectors(ubm, norm_stat_fn, num_thread=num_thread)
norm_iv.write(norm_iv_fn)

norm_mean, norm_cov = norm_iv.estimate_spectral_norm_stat1(1, 'sphNorm')

write_norm_hdf5([norm_mean, norm_cov], norm_fn)

norm_iv.spectral_norm_stat1(norm_mean[:1], norm_cov[:1])

fa = FactorAnalyser()
fa.plda(norm_iv, rank_plda, nb_iter=it_max_plda)
write_plda_hdf5([fa.mean, fa.F, numpy.zeros((rank_tv, 0)), fa.Sigma], plda_fn)