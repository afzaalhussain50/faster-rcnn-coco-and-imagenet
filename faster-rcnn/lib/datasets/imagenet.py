from __future__ import print_function
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.imagenet
# from imagenet_eval import voc_eval
import os, sys
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import subprocess
import pdb
import pickle
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


# Code from imagenet_eval.py reason need to build again
import xml.etree.ElementTree as ET
# import cPickle
def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(wnid_to_ind, class_to_ind,
             detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            print(annopath)
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            # commented by afzaal
            # cPickle.dump(recs, f)
            pickle.dump(recs, f)
            
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
#         print "Class name {}".format(classname)
#         for obj in recs[imagename]:
#             if wnid_to_ind[obj['name']] == class_to_ind[classname]:
#                 print "Object {} {}, {} ".format(obj['name'], classname, wnid_to_ind[obj['name']])
        
        R = [obj for obj in recs[imagename] if wnid_to_ind[obj['name']] == class_to_ind[classname]]
        bbox = np.array([x['bbox'] for x in R])
#         difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
#       npos = npos + sum(~difficult)
        npos = npos + len(R)
        class_recs[imagename] = {'bbox': bbox,
#                                  'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    # added by afzaal
    # print('lines', lines)
    if len(lines) <= 0:
        #  return rec, prec, ap
        return 0, 0, 0


    
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    print('sorted_ind',sorted_ind )
    print('BB', BB)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        # print("R['bbox']", R['bbox'])

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            # Code added by afzaal

            #                            m*n (width*height)
            #   ovthresh = min(0.5, _____________________ )
            #                          (m+10)*(n+10)

            # BBGTxmin = (BBGT[:, 0])
            # BBGTymin = (BBGT[:, 1])
            # BBGTxmax = (BBGT[:, 2])
            # BBGTymax = (BBGT[:, 3])
            # BBGTwidth = BBGTxmax - BBGTxmin
            # BBGTheight = BBGTymax - BBGTymin
            # print('BBGTwidth, BBGTheight', BBGTwidth, BBGTheight)
            # ovthresh = min(0.5, (BBGTwidth*BBGTheight) / ((BBGTwidth+10) * (BBGTheight+10)) )
            # print('ovthresh', ovthresh)

        if ovmax > ovthresh:
#             if not R['difficult'][jmax]:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap



class imagenet(imdb):
    def __init__(self, image_set, devkit_path, data_path):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = data_path
        synsets_image = sio.loadmat(os.path.join(self._devkit_path, 'data', 'meta_det.mat'))
        synsets_video = sio.loadmat(os.path.join(self._devkit_path, 'data', 'meta_vid.mat'))
        # self._classes_image = ('__background__',)
        # self._wnid_image = (0,)

        self._classes = ('__background__',)
        self._wnid = (0,)

        for i in xrange(200):
            self._classes = self._classes + (synsets_image['synsets'][0][i][2][0],)
            self._wnid = self._wnid + (synsets_image['synsets'][0][i][1][0],)

        # for i in xrange(30):
        #     self._classes = self._classes + (synsets_video['synsets'][0][i][2][0],)
        #     self._wnid = self._wnid + (synsets_video['synsets'][0][i][1][0],)

        self._wnid_to_ind = dict(zip(self._wnid, xrange(201)))
        self._class_to_ind = dict(zip(self._classes, xrange(201)))

        # self._wnid_to_ind = dict(zip(self._wnid, xrange(31)))
        # self._class_to_ind = dict(zip(self._classes, xrange(31)))

        #check for valid intersection between video and image classes
        # self._valid_image_flag = [0]*201
        #
        # for i in range(1,201):
        #     if self._wnid_image[i] in self._wnid_to_ind:
        #         self._valid_image_flag[i] = 1

        self._image_ext = ['.JPEG']

        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), 'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Data', self._image_set, index + self._image_ext[0])
        # print(image_path)
        assert os.path.exists(image_path), 'path does not exist: {}'.format(image_path)
        return image_path

    # Orignal code of this repositry this function is replaced by me with bottom-up-attention code
    # def _load_image_set_index(self):
    #     """
    #     Load the indexes listed in this dataset's image set file.
    #     """
    #     # Example path to image set file:
    #     # self._data_path + /ImageSets/val.txt
    #
    #     if self._image_set == 'train':
    #         image_set_file = os.path.join(self._data_path, 'ImageSets', 'trainr.txt')
    #         image_index = []
    #         if os.path.exists(image_set_file):
    #             f = open(image_set_file, 'r')
    #             data = f.read().split()
    #             for lines in data:
    #                 if lines != '':
    #                     image_index.append(lines)
    #             f.close()
    #             return image_index
    #
    #         for i in range(1,200):
    #             print(i)
    #             # Change by Afzaal
    #             # image_set_file = os.path.join(self._data_path, 'ImageSets', 'DET', 'train_' + str(i) + '.txt')
    #             image_set_file = os.path.join(self._data_path, 'ImageSets', 'DET', 'train_part_' + str(i) + '.txt')
    #
    #             with open(image_set_file) as f:
    #                 tmp_index = [x.strip() for x in f.readlines()]
    #                 vtmp_index = []
    #                 # if len(f.readlines()) <= 0:
    #                 #     continue
    #                 for line in tmp_index:
    #                     line = line.split(' ')
    #                     image_list = os.popen('ls ' + self._data_path + '/Data/DET/train/' + line[0] + '/*.JPEG').read().split()
    #                     tmp_list = []
    #                     for imgs in image_list:
    #                         tmp_list.append(imgs[:-5])
    #                     vtmp_index = vtmp_index + tmp_list
    #
    #             num_lines = len(vtmp_index)
    #             ids = np.random.permutation(num_lines)
    #             count = 0
    #             while count < 2000:
    #                 image_index.append(vtmp_index[ids[count % num_lines]])
    #                 count = count + 1
    #
    #         for i in range(1,201):
    #             if self._valid_image_flag[i] == 1:
    #                 image_set_file = os.path.join(self._data_path, 'ImageSets', 'train_pos_' + str(i) + '.txt')
    #                 with open(image_set_file) as f:
    #                     tmp_index = [x.strip() for x in f.readlines()]
    #                 num_lines = len(tmp_index)
    #                 ids = np.random.permutation(num_lines)
    #                 count = 0
    #                 while count < 2000:
    #                     image_index.append(tmp_index[ids[count % num_lines]])
    #                     count = count + 1
    #         image_set_file = os.path.join(self._data_path, 'ImageSets', 'trainr.txt')
    #         f = open(image_set_file, 'w')
    #         for lines in image_index:
    #             f.write(lines + '\n')
    #         f.close()
    #     else:
    #         image_set_file = os.path.join(self._data_path, 'ImageSets', 'val.txt')
    #         with open(image_set_file) as f:
    #             image_index = [x.strip() for x in f.readlines()]
    #     return image_index

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt

        if self._image_set == 'train':
            image_index = []
            # Change by Afzaal
            # for i in range(1, 201):
            for i in range(1, 201):
                # Change by Afzaal
                # if self._valid_image_flag[i] == 1:
                image_set_file = os.path.join(self._data_path, 'ImageSets', 'train_pos_' + str(i) + '.txt')
                with open(image_set_file) as f:
                    tmp_index = [x.strip() for x in f.readlines()]
                num_lines = len(tmp_index)
                ids = np.random.permutation(num_lines)
                count = 0
                while count < 2000:
                    image_index.append(tmp_index[ids[count % num_lines]])
                    count = count + 1

            # Change by Afzaal Commented this code
            # for i in range(1, 31):
            #     image_set_file = os.path.join(self._data_path, 'ImageSets', 'train_' + str(i) + '.txt')
            #     with open(image_set_file) as f:
            #         tmp_index = [x.strip() for x in f.readlines()]
            #     num_lines = len(tmp_index)
            #     ids = np.random.permutation(num_lines)
            #     count = 0
            #     while count < 2000:
            #         image_index.append(tmp_index[ids[count % num_lines]])
            #         count = count + 1
        else:
            image_set_file = os.path.join(self._data_path, 'ImageSets', 'val.txt')
            with open(image_set_file) as f:
                image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_imagenet_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of imagenet.
        """
        filename = os.path.join(self._data_path, 'Annotations', self._image_set, index + '.xml')

        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            # Change by Afzaal
            # cls = self._wnid_to_ind_image[
            #     str(get_data_from_tag(obj, "name")).lower().strip()]
            cls = self._wnid_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}


    def _write_imagenet_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        print("Use salt {}".format(use_salt))
        comp_id = 'comp4'
#         if use_salt:
#             comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
#             print 'Writing {} VOC results file'.format(cls)
            filename = self._get_imagenet_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        
        return comp_id
    
    def _get_imagenet_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = 'comp4' + '_{:s}.txt'
        print(filename)
        path = os.path.join(
            self._devkit_path, 'results',
            filename)
        return path

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'setenv(\'LC_ALL\',\'C\'); imagenet_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)
        
    def _do_python_eval(self, output_dir = 'output'):
        # Commented by afzaal
        # annopath = os.path.join(self._val_det_bbox, '{:s}.xml')
        annopath = os.path.join(self._data_path, 'Annotations', self._image_set, '{:s}.xml')
        print("Anno path {}".format(annopath))
        # Commented by afzaal
        # imagesetfile = os.path.join(
        #     self._devkit_path, "data/det_lists",
        #     self._image_set + '.txt')
        imagesetfile = os.path.join(self._data_path, 'ImageSets',  self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_imagenet_results_file_template().format(cls)
            print("File name {}".format(filename))
            rec, prec, ap = voc_eval(self._wnid_to_ind, self._class_to_ind,
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        i = 1
        for ap in aps:
            print('{}: {:.3f}'.format(self._classes[i], ap))
            i += 1
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir):
        self._comp_id = self._write_imagenet_results_file(all_boxes)
        self._do_python_eval(output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.imagenet('val', '')
    res = d.roidb
    from IPython import embed; embed()
