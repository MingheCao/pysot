from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, \
        VOTDataset, NFSDataset, VOTLTDataset
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, \
        EAOBenchmark, F1Benchmark
from toolkit.visualization.draw_success_precision import  draw_success_precision,draw_success_precision_paper

parser = argparse.ArgumentParser(description='tracking evaluation')
parser.add_argument('--tracker_path', '-p', type=str,
                    help='tracker result path')
parser.add_argument('--dataset', '-d', type=str,
                    help='dataset name')
parser.add_argument('--num', '-n', default=1, type=int,
                    help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t', default='',
                    type=str, help='tracker name')
parser.add_argument('--show_video_level', '-s', dest='show_video_level',
                    action='store_true')
parser.set_defaults(show_video_level=False)
args = parser.parse_args()


def main():
    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob(os.path.join(args.tracker_path,
                                 args.dataset,
                                 args.tracker_prefix+'*'))
    trackers = [os.path.basename(x) for x in trackers]

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    root = os.path.realpath(os.path.join(os.path.dirname(__file__),
                            '../testing_dataset'))
    root = os.path.join(root, args.dataset)

    dataset = UAVDataset(args.dataset, root)
    dataset.set_tracker(tracker_dir, trackers)
    benchmark = OPEBenchmark(dataset)
    success_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
            trackers), desc='eval success', total=len(trackers), ncols=100):
            success_ret.update(ret)
    precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
            trackers), desc='eval precision', total=len(trackers), ncols=100):
            precision_ret.update(ret)
    benchmark.show_result(success_ret, precision_ret,
            show_video_level=args.show_video_level)

    UGV_dict = {'210113_1': 'UGV01',
                '210113_2': 'UGV02',
                '210113_3': 'UGV03',
                '210113_4': 'UGV04',
                '210118_1': 'UGV05',
                '210118_2': 'UGV06',
                '210118_3': 'UGV07',
                '210119_1': 'UGV08',
                '210120_1': 'UGV09',
                '210120_2': 'UGV10',
                '210120_3': 'UGV11',
                '210120_4': 'UGV12',
                '210121_1': 'UGV13',
                '210121_2': 'UGV14',
                '210121_3': 'UGV15',
                '210121_4': 'UGV16',
                '210121_5': 'UGV17'}
    attr='210120_4'
    save_path = '/home/rislab/Workspace/pysot/rb_result/opefigs/'
    draw_success_precision_paper(success_ret, 'UGV', attr, UGV_dict[attr], precision_ret=precision_ret, bold_name=['Ours(Siamrpn)','Ours(Siamrpn++)'],save = True,save_path=save_path)
    # for nn in UGV_dict.keys():
    #     draw_success_precision_paper(success_ret,'UGV',nn,UGV_dict[nn],precision_ret=precision_ret,bold_name=['Ours(Siamrpn)','Ours(Siamrpn++)'],save = True,save_path=save_path)
    draw_success_precision_paper(success_ret,'UGV',list(success_ret['Ours(Siamrpn)'].keys()),'all',precision_ret=precision_ret,bold_name=['Ours(Siamrpn)','Ours(Siamrpn++)'],save = True,save_path=save_path)


if __name__ == '__main__':
    main()
