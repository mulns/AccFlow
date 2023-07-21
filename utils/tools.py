import imageio
import cv2
import numpy as np
import os
import torch.nn as nn
import torch
import pickle
import random
from shutil import rmtree
import time
import png
from PIL import Image
import re
from collections import Iterable
from matplotlib.colors import hsv_to_rgb
import yaml
from collections import OrderedDict
from easydict import EasyDict


class tools():

    class abstract_config():

        @property
        def to_dict(self):

            def dict_class(obj):
                temp = {}
                k = dir(obj)
                for name in k:
                    if not name.startswith('_') and name != 'to_dict':
                        value = getattr(obj, name)
                        if callable(value):
                            pass
                        else:
                            temp[name] = value
                return temp

            s_dict = dict_class(self)
            return s_dict

        @property
        def _key_list(self):
            k_list = list(self.to_dict.keys())
            return k_list

        def update(self, data: dict):

            t_key = list(data.keys())
            for i in self._key_list:
                if i in t_key:
                    setattr(self, i, data[i])
                    # print('set param ====  %s:   %s' % (i, data[i]))

        def __contains__(self, item):
            '''  use to check something in config '''
            if item in self._key_list:
                return True
            else:
                return False

        # todo 不知道这里写得对不对, 我是抄过来的
        def load_yml_file(self, yml_path):

            def ordered_yaml():
                """Support OrderedDict for yaml.

                Returns:
                    yaml Loader and Dumper.
                """
                try:
                    from yaml import CDumper as Dumper
                    from yaml import CLoader as Loader
                except ImportError:
                    from yaml import Dumper, Loader

                _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

                def dict_representer(dumper, data):
                    return dumper.represent_dict(data.items())

                def dict_constructor(loader, node):
                    return OrderedDict(loader.construct_pairs(node))

                Dumper.add_representer(OrderedDict, dict_representer)
                Loader.add_constructor(_mapping_tag, dict_constructor)
                return Loader, Dumper

            # parse yml to dict
            with open(yml_path, mode='r') as f:
                opt = yaml.load(f, Loader=ordered_yaml()[0])
            opt = EasyDict(opt)
            return opt

        # todo 不知道这里写得对不对, 我是抄过来的, 不知道这样存下来, 能不能原样这样读取出来
        def save_yml_file(self, yml_path):
            conf_dict = self.to_dict

        @classmethod
        def __demo(cls):

            class temp(tools.abstract_config):

                def __init__(self, **kwargs):
                    self.if_gpu = True
                    self.eval_batch_size = 1
                    self.eval_name = 'flyingchairs'
                    self.eval_datatype = 'nori'  # or base
                    self.if_print_process = False

                    self.update(kwargs)

            a = temp(eval_batch_size=8,
                     eval_name='flyingchairs',
                     eval_datatype='nori',
                     if_print_process=False)

    class abs_database():

        def sample(self, index):
            pass

        def __len__(self):
            return 0

    class abstract_model(nn.Module):

        def save_model(self, save_path):
            torch.save(self.state_dict(), save_path)

        def load_model(self, load_path, if_relax=False, if_print=True):
            if if_print:
                print('loading protrained model from %s' % load_path)
            if if_relax:
                model_dict = self.state_dict()
                pretrained_dict = torch.load(load_path)
                # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                pretrained_dict_v2 = {}
                for k, v in pretrained_dict.items():
                    if k in model_dict:
                        if v.shape == model_dict[k].shape:
                            pretrained_dict_v2[k] = v
                model_dict.update(pretrained_dict_v2)
                self.load_state_dict(model_dict)
            else:
                self.load_state_dict(torch.load(load_path))

        def load_from_model(self, model: nn.Module, if_relax=False):
            if if_relax:
                model_dict = self.state_dict()
                pretrained_dict = model.state_dict()
                # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                pretrained_dict_v2 = {}
                for k, v in pretrained_dict.items():
                    if k in model_dict:
                        if v.shape == model_dict[k].shape:
                            pretrained_dict_v2[k] = v
                model_dict.update(pretrained_dict_v2)
                self.load_state_dict(model_dict)
            else:
                self.load_state_dict(model.state_dict())

        def choose_gpu(self, gpu_opt=None):
            # choose gpu
            if gpu_opt is None:
                # gpu=0
                model = self.cuda()
                # torch.cuda.set_device(gpu)
                # model.cuda(gpu)
                # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
                # print('torch.cuda.device_count()  ',torch.cuda.device_count())
                # model=torch.nn.parallel.DistributedDataParallel(model,device_ids=range(torch.cuda.device_count()))
                model = torch.nn.DataParallel(
                    model, device_ids=list(range(
                        torch.cuda.device_count())))  # multi gpu
            elif gpu_opt == 0:
                model = self.cuda()
            else:
                if type(gpu_opt) != int:
                    raise ValueError('wrong gpu config:  %s' % (str(gpu_opt)))
                torch.cuda.set_device(gpu_opt)
                model = self.cuda(gpu_opt)
            return model

        @classmethod
        def save_model_gpu(cls, model, path):
            name_dataparallel = torch.nn.DataParallel.__name__
            if type(model).__name__ == name_dataparallel:
                model = model.module
            model.save_model(path)

    class AverageMeter():

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, num):
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

    class Avg_meter_ls():

        def __init__(self):
            self.data_ls = {}
            self.short_name_ls = {}

        def update(self, name, val, num, short_name=None):
            if name not in self.data_ls.keys():
                self.data_ls[name] = tools.AverageMeter()
                if short_name is None:
                    short_name = name
                self.short_name_ls[name] = short_name
            self.data_ls[name].update(val=val, num=num)

        def print_loss(self, name):
            a = ' %s %.4f(%.4f)' % (self.short_name_ls[name],
                                    self.data_ls[name].val,
                                    self.data_ls[name].avg)
            return a

        def print_avg_loss(self, name):
            a = ' %s: %.4f' % (self.short_name_ls[name],
                               self.data_ls[name].avg)
            return a

        def print_all_losses(self):
            a = ''
            for i in sorted(self.data_ls.keys()):
                a += ' %s %.4f(%.4f)' % (self.short_name_ls[i],
                                         self.data_ls[i].val,
                                         self.data_ls[i].avg)
            return a

        def print_all_losses_final(self):
            a = ''
            for i in sorted(self.data_ls.keys()):
                a += ' %s=%.4f' % (self.short_name_ls[i], self.data_ls[i].avg)
            return a

        def get_all_losses_final(self):
            a = {}
            for i in sorted(self.data_ls.keys()):
                a[i] = self.data_ls[i].avg
            return a

        def reset(self):
            for name in self.data_ls.keys():
                self.data_ls[name].reset()

    class TimeClock():

        def __init__(self):
            self.st = 0
            self.en = 0
            self.start_flag = False

        def start(self):
            self.reset()
            self.start_flag = True
            self.st = time.time()

        def reset(self):
            self.start_flag = False
            self.st = 0
            self.en = 0

        def end(self):
            self.en = time.time()

        def get_during(self):
            return self.en - self.st

    # 研究一下图像加字体展示结果
    class Text_img():

        def __init__(self, **kwargs):
            self.font = 'simplex'
            self.my_font_type = 'black_white'
            self.__update(kwargs)
            self.font_ls = {
                'simplex': cv2.FONT_HERSHEY_SIMPLEX,
                'plain': cv2.FONT_HERSHEY_PLAIN,
                'complex': cv2.FONT_HERSHEY_COMPLEX,
                'trplex': cv2.FONT_HERSHEY_TRIPLEX,
                # 'complex_small': cv2.FONT_HERSHEY_COMPLEX_SMALL,
                'italic': cv2.FONT_ITALIC,
            }
            self.my_font_type_ls = {
                'black_white': self._black_white,
            }
            self.show_func = self.my_font_type_ls[self.my_font_type]

        def __update(self, data: dict):

            def dict_class(obj):
                temp = {}
                k = dir(obj)
                for name in k:
                    if not name.startswith('_'):
                        value = getattr(obj, name)
                        if callable(value):
                            pass
                        else:
                            temp[name] = value
                return temp

            s_dict = dict_class(self)
            k_list = list(s_dict.keys())
            t_key = list(data.keys())
            for i in k_list:
                if i in t_key:
                    setattr(self, i, data[i])
                    # print('set param ====  %s:   %s' % (i, data[i]))

        def _black_white(self, img, text, scale, row=0):
            # params
            color_1 = (10, 10, 10)
            thick_1 = 5
            color_2 = (255, 255, 255)
            thick_2 = 2

            # get position: Bottom-left
            t_w, t_h, t_inter = self._check_text_size(text=text,
                                                      scale=scale,
                                                      thick=thick_1)
            pw = t_inter
            ph = t_h + t_inter + row * (t_h + t_inter)

            # put text
            img_ = img.copy()
            img_ = cv2.putText(img_,
                               text, (pw, ph),
                               fontFace=self.font_ls[self.font],
                               fontScale=scale,
                               color=color_1,
                               thickness=thick_1)
            img_ = cv2.putText(img_,
                               text, (pw, ph),
                               fontFace=self.font_ls[self.font],
                               fontScale=scale,
                               color=color_2,
                               thickness=thick_2)
            return img_

        def _check_text_size(self, text: str, scale=1, thick=1):
            textSize, baseline = cv2.getTextSize(text, self.font_ls[self.font],
                                                 scale, thick)
            twidth, theight = textSize
            return twidth, theight, baseline // 2

        def put_text(self, img, text=None, scale=1):
            if text is not None:
                if type(text) == str:
                    img = self.show_func(img, text, scale, 0)
                elif isinstance(text, Iterable):
                    for i, t in enumerate(text):
                        img = self.show_func(img, t, scale, i)
            return img

        def draw_cross(self,
                       img,
                       point_wh,
                       cross_length=5,
                       color=(0, 0, 255)):  #
            thick = cross_length // 2
            new_img = img.copy()
            new_img = cv2.line(new_img,
                               (point_wh[0] - cross_length, point_wh[1]),
                               (point_wh[0] + cross_length, point_wh[1]),
                               color, thick)
            new_img = cv2.line(new_img,
                               (point_wh[0], point_wh[1] - cross_length),
                               (point_wh[0], point_wh[1] + cross_length),
                               color, thick)
            return new_img

        def draw_cross_black_white(self, img, point_wh, cross_length=5):  #
            if cross_length <= 5:
                cross_length = 5
            thick = cross_length // 2
            new_img = img.copy()
            new_img = cv2.line(new_img,
                               (point_wh[0] - cross_length, point_wh[1]),
                               (point_wh[0] + cross_length, point_wh[1]),
                               (0, 0, 0), thick)
            new_img = cv2.line(new_img,
                               (point_wh[0], point_wh[1] - cross_length),
                               (point_wh[0], point_wh[1] + cross_length),
                               (0, 0, 0), thick)
            new_img = cv2.line(new_img,
                               (point_wh[0] - cross_length, point_wh[1]),
                               (point_wh[0] + cross_length, point_wh[1]),
                               (250, 250, 250), thick // 2)
            new_img = cv2.line(new_img,
                               (point_wh[0], point_wh[1] - cross_length),
                               (point_wh[0], point_wh[1] + cross_length),
                               (250, 250, 250), thick // 2)
            return new_img

        def draw_x(self, img, point_wh, cross_length=5, color=(0, 0, 255)):
            thick = cross_length // 2
            new_img = img.copy()
            new_img = cv2.line(
                new_img,
                (point_wh[0] - cross_length, point_wh[1] - cross_length),
                (point_wh[0] + cross_length, point_wh[1] + cross_length),
                color, thick)
            new_img = cv2.line(
                new_img,
                (point_wh[0] - cross_length, point_wh[1] + cross_length),
                (point_wh[0] + cross_length, point_wh[1] - cross_length),
                color, thick)
            return new_img

        def draw_x_black_white(self, img, point_wh, cross_length=5):
            if cross_length <= 5:
                cross_length = 5
            thick = cross_length // 2
            new_img = img.copy()
            new_img = cv2.line(
                new_img,
                (point_wh[0] - cross_length, point_wh[1] - cross_length),
                (point_wh[0] + cross_length, point_wh[1] + cross_length),
                (0, 0, 0), thick)
            new_img = cv2.line(
                new_img,
                (point_wh[0] - cross_length, point_wh[1] + cross_length),
                (point_wh[0] + cross_length, point_wh[1] - cross_length),
                (0, 0, 0), thick)
            new_img = cv2.line(
                new_img,
                (point_wh[0] - cross_length, point_wh[1] - cross_length),
                (point_wh[0] + cross_length, point_wh[1] + cross_length),
                (250, 250, 250), thick // 2)
            new_img = cv2.line(
                new_img,
                (point_wh[0] - cross_length, point_wh[1] + cross_length),
                (point_wh[0] + cross_length, point_wh[1] - cross_length),
                (250, 250, 250), thick // 2)
            return new_img

        def show_img_dict(self, **kwargs):
            for i in kwargs.keys():
                img = self.put_text(kwargs[i], text=i)
                cv2.imshow(i, img)
            cv2.waitKey()

        def demo(self):
            im = np.ones((500, 500, 3), dtype='uint8') * 50
            imshow = self.put_text(im,
                                   text=list(
                                       'demo show sample text'.split(' ')),
                                   scale=1)
            cv2.imshow('im', imshow)
            cv2.waitKey()

    @classmethod
    def clear(cls):
        os.system("clear")  # 清屏

    @classmethod
    def random_flag(cls, threshold_0_1=0.5):
        a = random.random() < threshold_0_1
        return a


class file_tools():

    class Nori_tools():
        # need id, shape and type to read nori data
        class Fetcher():

            def __init__(self):
                self.fetcher = nori.Fetcher()

            def get(self,
                    id=0,
                    shape=(320, 640, 3),
                    data_type=np.uint8,
                    **kwargs):
                byte_data = self.fetcher.get(id)
                img = np.frombuffer(byte_data, data_type)
                img = np.reshape(img, shape)
                return img

        # filepath: local dir，target: target dir on oss, here I set my oss path: 's3://luokunming/Optical_Flow_all', you may change to yours
        @classmethod
        def save_file_to_oss(cls, file_path, target_path):
            # os.system('alias oss="aws --endpoint-url=http://oss.hh-b.brainpp.cn s3"')
            # target_path = os.path.join('s3://luokunming/Optical_Flow_all/datasets', target, os.path.basename(file_path))
            # target_path = 's3://luokunming/Optical_Flow_all/datasets_v2/%s/%s' % (target, os.path.basename(file_path))
            os.system(
                'aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 cp %s %s' %
                (file_path, target_path))
            return target_path

        # output command line to upload files to oss
        @classmethod
        def save_filedir_to_oss(cls, file_dir_path, target):
            # os.system('alias oss="aws --endpoint-url=http://oss.hh-b.brainpp.cn s3"')
            # target_path = os.path.join('s3://luokunming/Optical_Flow_all/datasets', target, os.path.basename(file_dir_path))
            target_path = 's3://luokunming/Optical_Flow_all/datasets_v2/%s/%s' % (
                target, os.path.basename(file_dir_path))
            os.system(
                'aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 sync %s %s' %
                (file_dir_path, target_path))
            return target_path

        # speed up nori
        @classmethod
        def nori_speedup_file(cls, file_path):
            '''
                    nori speedup file_path --on --replica 6
            sintel train数据加速
            nori speedup s3://luokunming/Optical_Flow_all/datasets/Sintel_training_set/traning_set.nori --on --replica 6
            sintel raw数据加速
            nori speedup s3://luokunming/Optical_Flow_all/datasets/Sintel_raw_dataset/traning_set.nori--on --replica 6
            '''
            print(file_path + ' speed up!!!!!!!!!!!')
            os.system('nori speedup ' + file_path + ' --on' + ' --replica 6')

        class Nori_saver():
            '''
            how to use：
            1. initialize an instance by giving out_nori_path, which is the dir to save nori file
            2. use 'start' to do some initialization
            3. prepare your image(numpy) and name(str)
            4. use 'end' to finish saving nori file (note that: if you do not use this function, nori file cannot be correctly saved)
            5. you can optionally speed up your nori file

            '''

            recorder_nr = None

            def __init__(self, out_nori_path='', if_remote=False):
                self.out_nori_path = out_nori_path
                self.if_remote = if_remote

            def start(self):
                if self.if_remote:
                    self.recorder_nr = nori.remotewriteopen(self.out_nori_path)
                else:
                    pass
                self.recorder_nr = nori.open(self.out_nori_path, "w")
                print('=' * 3 +
                      ' start making nori file: %s' % self.out_nori_path)

            def end(self):
                self.recorder_nr.close()

            def put_file(self, img, name):
                img_temp = img.tobytes()
                data_id1 = self.recorder_nr.put(img_temp, filename=name)
                data = {
                    'id': data_id1,
                    'shape': img.shape,
                    'data_type': img.dtype,
                    'name': name
                }
                return data

            def upload_oss_and_speedup(self, dataset_name):
                oss_path = file_tools.Nori_tools.save_filedir_to_oss(
                    self.out_nori_path, dataset_name)
                file_tools.Nori_tools.nori_speedup_file(oss_path)

            def nori_speedup(self):
                pcmd = 'nori speedup ' + self.out_nori_path + ' --on' + ' --replica 2'
                print('===')
                print(pcmd)
                print('===')
                os.system(pcmd)

        @classmethod
        def demo(cls):
            # === save nori files direcctly to oss
            nori_path = 's3://luokunming/temp.nori'
            data_ls = [
                np.zeros((10, 10, 3)),
            ]
            nsaver = file_tools.Nori_tools.Nori_saver(out_nori_path=nori_path,
                                                      if_remote=True)
            nsaver.start()
            new_data_ls = []
            for ind, data in enumerate(data_ls):
                name = '%s' % ind
                nori_id = nsaver.put_file(img=data, name=name)
                new_data_ls.append(nori_id)
            nsaver.end()
            # === end save nori file, and speed up
            nsaver.nori_speedup()
            # === wait a minute to speed up nori file
            import time
            for i in range(10):
                print('waite a minute...[%i|10]' % i)
                time.sleep(6)

            # === test read nori files
            nori_fetcher = file_tools.Nori_tools.Fetcher()
            for ind in new_data_ls:
                data = nori_fetcher.get(**ind)
                tensor_tools.check_tensor_np(data, ind['name'])

    class npz_saver():

        @classmethod
        def save_npz(cls, files, npz_save_path):
            np.savez(npz_save_path, files=[files, 0])

        @classmethod
        def load_npz(cls, npz_save_path):
            with np.load(npz_save_path) as fin:
                files = fin['files']
                files = list(files)
                return files[0]

    class pickle_saver():

        @classmethod
        def save_pickle(cls, files, file_path):
            with open(file_path, 'wb') as data:
                pickle.dump(files, data)

        @classmethod
        def load_picke(cls, file_path):
            with open(file_path, 'rb') as data:
                data = pickle.load(data)
            return data

    class txt_read_write():

        @classmethod
        def read(cls, path):
            with open(path, "r") as f:
                data = f.readlines()
            return data

        @classmethod
        def write(cls, path, data_ls):
            file_write_obj = open(path, 'a')
            for i in data_ls:
                file_write_obj.writelines(i)
            file_write_obj.close()

        @classmethod
        def demo(cls):
            txt_path = r'E:\research\unsupervised_optical_flow\projects\Ric-master\Ric-master\data\MPI-Sintel\frame_0001_match.txt'
            data = file_tools.txt_read_write.read(txt_path)
            print(' ')
            write_txt_path = txt_path = r'E:\research\unsupervised_optical_flow\projects\Ric-master\Ric-master\data\MPI-Sintel\temp.txt'
            file_tools.txt_read_write.write(write_txt_path, data[:10])
            print(' ')

    class flow_read_write():

        @classmethod
        def write_flow_png(cls, filename, uv, v=None, mask=None):

            if v is None:
                assert (uv.ndim == 3)
                assert (uv.shape[2] == 2)
                u = uv[:, :, 0]
                v = uv[:, :, 1]
            else:
                u = uv

            assert (u.shape == v.shape)

            height_img, width_img = u.shape
            if mask is None:
                valid_mask = np.ones([height_img, width_img], dtype=np.uint16)
            else:
                valid_mask = mask

            flow_u = np.clip((u * 64 + 2**15), 0.0, 65535.0).astype(np.uint16)
            flow_v = np.clip((v * 64 + 2**15), 0.0, 65535.0).astype(np.uint16)

            output = np.stack((flow_u, flow_v, valid_mask), axis=-1)

            with open(filename, 'wb') as f:
                # writer = png.Writer(width=width_img, height=height_img, bitdepth=16)
                # temp = np.reshape(output, (-1, width_img * 3))
                # writer.write(f, temp)

                png_writer = png.Writer(width=width_img,
                                        height=height_img,
                                        bitdepth=16,
                                        compression=3,
                                        greyscale=False)
                # png_writer.write_array(f, output)
                temp = np.reshape(output, (-1, width_img * 3))
                png_writer.write(f, temp)

        @classmethod
        def write_kitti_png_file(cls, flow_fn, flow_data, mask_data=None):
            flow_img = np.zeros((flow_data.shape[0], flow_data.shape[1], 3),
                                dtype=np.uint16)
            if mask_data is None:
                mask_data = np.ones([flow_data.shape[0], flow_data.shape[1]],
                                    dtype=np.uint16)
            flow_img[:, :,
                     2] = (flow_data[:, :, 0] * 64.0 + 2**15).astype(np.uint16)
            flow_img[:, :,
                     1] = (flow_data[:, :, 1] * 64.0 + 2**15).astype(np.uint16)
            flow_img[:, :, 0] = mask_data[:, :]
            cv2.imwrite(flow_fn, flow_img)

        @classmethod
        def read_flo(cls, filename):
            with open(filename, 'rb') as f:
                magic = np.fromfile(f, np.float32, count=1)
                if 202021.25 != magic:
                    print('Magic number incorrect. Invalid .flo file')
                else:
                    w = np.fromfile(f, np.int32, count=1)
                    h = np.fromfile(f, np.int32, count=1)
                    data = np.fromfile(f, np.float32, count=int(2 * w * h))
                    # Reshape data into 3D array (columns, rows, bands)
                    data2D = np.resize(data, (h[0], w[0], 2))
                    return data2D

        @classmethod
        def write_flo(cls, flow, filename):
            """
            write optical flow in Middlebury .flo format
            :param flow: optical flow map
            :param filename: optical flow file path to be saved
            :return: None
            """
            f = open(filename, 'wb')
            magic = np.array([202021.25], dtype=np.float32)
            (height, width) = flow.shape[0:2]
            w = np.array([width], dtype=np.int32)
            h = np.array([height], dtype=np.int32)
            magic.tofile(f)
            w.tofile(f)
            h.tofile(f)
            flow.tofile(f)
            f.close()

    @classmethod
    def check_dir(cls, path):
        if not os.path.exists(path):
            os.makedirs(path)

    @classmethod
    def tryremove(cls, name, file=False):
        try:
            if file:
                os.remove(name)
            else:
                rmtree(name)
        except OSError:
            pass


class tensor_tools():

    @classmethod
    def torch_warp_mask(cls, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """

        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
        mask = torch.autograd.Variable(torch.ones(x.size()))
        if x.is_cuda:
            mask = mask.cuda()
        mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        output = output * mask
        # # nchw->>>nhwc
        # if x.is_cuda:
        #     output = output.cpu()
        # output_im = output.numpy()
        # output_im = np.transpose(output_im, (0, 2, 3, 1))
        # output_im = np.squeeze(output_im)
        return output, mask

    @classmethod
    def torch_warp(cls, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """

        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        # tools.check_tensor(x, 'x')
        # tools.check_tensor(vgrid, 'vgrid')
        output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
        # mask = torch.autograd.Variable(torch.ones(x.size()))
        # if x.is_cuda:
        #     mask = mask.cuda()
        # mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
        #
        # mask[mask < 0.9999] = 0
        # mask[mask > 0] = 1
        # output = output * mask
        # # nchw->>>nhwc
        # if x.is_cuda:
        #     output = output.cpu()
        # output_im = output.numpy()
        # output_im = np.transpose(output_im, (0, 2, 3, 1))
        # output_im = np.squeeze(output_im)
        return output

    @classmethod
    def torch_warp_boundary(cls, x, flo, start_point):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        start_point: [B,2,1,1]
        """

        _, _, Hx, Wx = x.size()
        B, C, H, W = flo.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flo + start_point

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(Wx - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(Hx - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        # tools.check_tensor(x, 'x')
        # tools.check_tensor(vgrid, 'vgrid')
        output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
        # mask = torch.autograd.Variable(torch.ones(x.size()))
        # if x.is_cuda:
        #     mask = mask.cuda()
        # mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
        #
        # mask[mask < 0.9999] = 0
        # mask[mask > 0] = 1
        # output = output * mask
        # # nchw->>>nhwc
        # if x.is_cuda:
        #     output = output.cpu()
        # output_im = output.numpy()
        # output_im = np.transpose(output_im, (0, 2, 3, 1))
        # output_im = np.squeeze(output_im)
        return output

    @classmethod
    def create_gif(cls, image_list, gif_name, duration=0.5):
        frames = []
        for image_name in image_list:
            frames.append(image_name)
        imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
        return

    @classmethod
    def warp_cv2(cls, img_prev, flow):
        # calculate mat
        w = int(img_prev.shape[1])
        h = int(img_prev.shape[0])
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.float32(np.dstack([x_coords, y_coords]))
        pixel_map = coords + flow
        new_frame = cv2.remap(img_prev, pixel_map, None, cv2.INTER_LINEAR)
        return new_frame

    @classmethod
    def flow_to_image_dmax(cls, flow, display=False):
        """

        :param flow: H,W,2
        :param display:
        :return: H,W,3
        """

        def compute_color(u, v):

            def make_color_wheel():
                """
                Generate color wheel according Middlebury color code
                :return: Color wheel
                """
                RY = 15
                YG = 6
                GC = 4
                CB = 11
                BM = 13
                MR = 6

                ncols = RY + YG + GC + CB + BM + MR

                colorwheel = np.zeros([ncols, 3])

                col = 0

                # RY
                colorwheel[0:RY, 0] = 255
                colorwheel[0:RY, 1] = np.transpose(
                    np.floor(255 * np.arange(0, RY) / RY))
                col += RY

                # YG
                colorwheel[col:col + YG, 0] = 255 - np.transpose(
                    np.floor(255 * np.arange(0, YG) / YG))
                colorwheel[col:col + YG, 1] = 255
                col += YG

                # GC
                colorwheel[col:col + GC, 1] = 255
                colorwheel[col:col + GC, 2] = np.transpose(
                    np.floor(255 * np.arange(0, GC) / GC))
                col += GC

                # CB
                colorwheel[col:col + CB, 1] = 255 - np.transpose(
                    np.floor(255 * np.arange(0, CB) / CB))
                colorwheel[col:col + CB, 2] = 255
                col += CB

                # BM
                colorwheel[col:col + BM, 2] = 255
                colorwheel[col:col + BM, 0] = np.transpose(
                    np.floor(255 * np.arange(0, BM) / BM))
                col += +BM

                # MR
                colorwheel[col:col + MR, 2] = 255 - np.transpose(
                    np.floor(255 * np.arange(0, MR) / MR))
                colorwheel[col:col + MR, 0] = 255

                return colorwheel

            """
            compute optical flow color map
            :param u: optical flow horizontal map
            :param v: optical flow vertical map
            :return: optical flow in color code
            """
            [h, w] = u.shape
            img = np.zeros([h, w, 3])
            nanIdx = np.isnan(u) | np.isnan(v)
            u[nanIdx] = 0
            v[nanIdx] = 0

            colorwheel = make_color_wheel()
            ncols = np.size(colorwheel, 0)

            rad = np.sqrt(u**2 + v**2)

            a = np.arctan2(-v, -u) / np.pi

            fk = (a + 1) / 2 * (ncols - 1) + 1

            k0 = np.floor(fk).astype(int)

            k1 = k0 + 1
            k1[k1 == ncols + 1] = 1
            f = fk - k0

            for i in range(0, np.size(colorwheel, 1)):
                tmp = colorwheel[:, i]
                col0 = tmp[k0 - 1] / 255
                col1 = tmp[k1 - 1] / 255
                col = (1 - f) * col0 + f * col1

                idx = rad <= 1
                col[idx] = 1 - rad[idx] * (1 - col[idx])
                notidx = np.logical_not(idx)

                col[notidx] *= 0.75
                img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

            return img

        UNKNOWN_FLOW_THRESH = 1e7
        """
        Convert flow into middlebury color code image
        :param flow: optical flow map
        :return: optical flow image in middlebury color
        """
        u = flow[:, :, 0]
        v = flow[:, :, 1]

        maxu = -999.
        maxv = -999.
        minu = 999.
        minv = 999.

        idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) >
                                                      UNKNOWN_FLOW_THRESH)
        u[idxUnknow] = 0
        v[idxUnknow] = 0

        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))

        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))

        rad = np.sqrt(u**2 + v**2)
        maxrad = max(-1, np.max(rad))

        if display:
            print(
                "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f"
                % (maxrad, minu, maxu, minv, maxv))

        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)

        img = compute_color(u, v)

        idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
        img[idx] = 0

        return np.uint8(img)

    @classmethod
    def flow_to_image_ndmax(cls, flow, max_flow=256):
        # flow shape (H, W, C)
        if max_flow is not None:
            max_flow = max(max_flow, 1.)
        else:
            max_flow = np.max(flow)

        n = 8
        u, v = flow[:, :, 0], flow[:, :, 1]
        mag = np.sqrt(np.square(u) + np.square(v))
        angle = np.arctan2(v, u)
        im_h = np.mod(angle / (2 * np.pi) + 1, 1)
        im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
        im_v = np.clip(n - im_s, a_min=0, a_max=1)
        im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
        return (im * 255).astype(np.uint8)

    @classmethod
    def flow_error_image_np(cls,
                            flow_pred,
                            flow_gt,
                            mask_occ,
                            mask_noc=None,
                            log_colors=True):
        """Visualize the error between two flows as 3-channel color image.
        Adapted from the KITTI C++ devkit.
        Args:
            flow_pred: prediction flow of shape [ height, width, 2].
            flow_gt: ground truth
            mask_occ: flow validity mask of shape [num_batch, height, width, 1].
                Equals 1 at (occluded and non-occluded) valid pixels.
            mask_noc: Is 1 only at valid pixels which are not occluded.
        """
        # mask_noc = tf.ones(tf.shape(mask_occ)) if mask_noc is None else mask_noc
        mask_noc = np.ones(mask_occ.shape) if mask_noc is None else mask_noc
        diff_sq = (flow_pred - flow_gt)**2
        # diff = tf.sqrt(tf.reduce_sum(diff_sq, [3], keep_dims=True))
        diff = np.sqrt(np.sum(diff_sq, axis=2, keepdims=True))
        if log_colors:
            height, width, _ = flow_pred.shape
            # num_batch, height, width, _ = tf.unstack(tf.shape(flow_1))
            colormap = [[0, 0.0625, 49, 54,
                         149], [0.0625, 0.125, 69, 117, 180],
                        [0.125, 0.25, 116, 173, 209],
                        [0.25, 0.5, 171, 217, 233], [0.5, 1, 224, 243, 248],
                        [1, 2, 254, 224, 144], [2, 4, 253, 174, 97],
                        [4, 8, 244, 109, 67], [8, 16, 215, 48, 39],
                        [16, 1000000000.0, 165, 0, 38]]
            colormap = np.asarray(colormap, dtype=np.float32)
            colormap[:, 2:5] = colormap[:, 2:5] / 255
            # mag = tf.sqrt(tf.reduce_sum(tf.square(flow_2), 3, keep_dims=True))
            tempp = np.square(flow_gt)
            # temp = np.sum(tempp, axis=2, keep_dims=True)
            # mag = np.sqrt(temp)
            mag = np.sqrt(np.sum(tempp, axis=2, keepdims=True))
            # error = tf.minimum(diff / 3, 20 * diff / mag)
            error = np.minimum(diff / 3, 20 * diff / (mag + 1e-7))
            im = np.zeros([height, width, 3])
            for i in range(colormap.shape[0]):
                colors = colormap[i, :]
                cond = np.logical_and(np.greater_equal(error, colors[0]),
                                      np.less(error, colors[1]))
                # temp=np.tile(cond, [1, 1, 3])
                im = np.where(np.tile(cond, [1, 1, 3]),
                              np.ones([height, width, 1]) * colors[2:5], im)
            # temp=np.cast(mask_noc, np.bool)
            # im = np.where(np.tile(np.cast(mask_noc, np.bool), [1, 1, 3]), im, im * 0.5)
            im = np.where(np.tile(mask_noc == 1, [1, 1, 3]), im, im * 0.5)
            im = im * mask_occ
        else:
            error = (np.minimum(diff, 5) / 5) * mask_occ
            im_r = error  # errors in occluded areas will be red
            im_g = error * mask_noc
            im_b = error * mask_noc
            im = np.concatenate([im_r, im_g, im_b], axis=2)
            # im = np.concatenate(axis=2, values=[im_r, im_g, im_b])
        return im[:, :, ::-1]

    @classmethod
    def compute_model_size(cls, model, *args):
        from thop import profile
        flops, params = profile(model, inputs=args, verbose=False)
        print('flops: %.3f G, params: %.3f M' %
              (flops / 1000 / 1000 / 1000, params / 1000 / 1000))

    @classmethod
    def count_parameters(cls, model):
        a = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return a

    @classmethod
    def im_norm(cls, img):
        eps = 1e-6
        a = np.max(img)
        b = np.min(img)
        if a - b <= 0:
            img = (img - b) / (a - b + eps)
        else:
            img = (img - b) / (a - b)
        img = img * 255
        img = img.astype('uint8')
        return img

    @classmethod
    def check_tensor(cls, data, name, print_data=False, print_in_txt=None):
        if data.is_cuda:
            temp = data.detach().cpu().numpy()
        else:
            temp = data.detach().numpy()
        a = len(name)
        name_ = name + ' ' * 100
        name_ = name_[0:max(a, 10)]
        print_str = '%s, %s, %s, %s,%s,%s,%s,%s' % (
            name_, temp.shape, data.dtype, ' max:%.2f' % np.max(temp),
            ' min:%.2f' % np.min(temp), ' mean:%.2f' % np.mean(temp),
            ' sum:%.2f' % np.sum(temp), data.device)
        if print_in_txt is None:
            print(print_str)
        else:
            print(print_str, file=print_in_txt)
        if print_data:
            print(temp)
        return print_str

    @classmethod
    def check_tensor_np(cls, data, name, print_data=False, print_in_txt=None):
        temp = data
        a = len(name)
        name_ = name + ' ' * 100
        name_ = name_[0:max(a, 10)]
        print_str = '%s, %s, %s, %s,%s,%s,%s' % (
            name_, temp.shape, data.dtype, ' max:%.2f' % np.max(temp),
            ' min:%.2f' % np.min(temp), ' mean:%.2f' % np.mean(temp),
            ' sum:%.2f' % np.sum(temp))
        if print_in_txt is None:
            print(print_str)
        else:
            print(print_str, file=print_in_txt)
        if print_data:
            print(temp)
        return print_str


class frame_utils():
    '''  borrowed from RAFT '''
    TAG_CHAR = np.array([202021.25], np.float32)

    @classmethod
    def readFlow(cls, fn):
        """ Read .flo file in Middlebury format"""
        # Code adapted from:
        # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

        # WARNING: this will work on little-endian architectures (eg Intel x86) only!
        # print 'fn = %s'%(fn)
        with open(fn, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
                return None
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                # print 'Reading %d x %d flo file\n' % (w, h)
                data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
                # Reshape data into 3D array (columns, rows, bands)
                # The reshape here is for visualization, the original code is (w,h,2)
                return np.resize(data, (int(h), int(w), 2))

    @classmethod
    def readPFM(cls, file):
        file = open(file, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data

    @classmethod
    def writeFlow(cls, filename, uv, v=None):
        """ Write optical flow to file.

        If v is None, uv is assumed to contain both u and v channels,
        stacked in depth.
        Original code by Deqing Sun, adapted from Daniel Scharstein.
        """
        nBands = 2

        if v is None:
            assert (uv.ndim == 3)
            assert (uv.shape[2] == 2)
            u = uv[:, :, 0]
            v = uv[:, :, 1]
        else:
            u = uv

        assert (u.shape == v.shape)
        height, width = u.shape
        f = open(filename, 'wb')
        # write the header
        f.write(cls.TAG_CHAR)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        # arrange into matrix form
        tmp = np.zeros((height, width * nBands))
        tmp[:, np.arange(width) * 2] = u
        tmp[:, np.arange(width) * 2 + 1] = v
        tmp.astype(np.float32).tofile(f)
        f.close()

    @classmethod
    def readFlowKITTI(cls, filename):
        flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        flow = flow[:, :, ::-1].astype(np.float32)
        flow, valid = flow[:, :, :2], flow[:, :, 2]
        flow = (flow - 2**15) / 64.0
        return flow, valid

    @classmethod
    def read_png_flow(cls, fpath):
        """
        Read KITTI optical flow, returns u,v,valid mask

        """

        R = png.Reader(fpath)
        width, height, data, _ = R.asDirect()
        # This only worked with python2.
        # I = np.array(map(lambda x:x,data)).reshape((height,width,3))
        gt = np.array([x for x in data]).reshape((height, width, 3))
        flow = gt[:, :, 0:2]
        flow = (flow.astype('float64') - 2**15) / 64.0
        flow = flow.astype(np.float)
        mask = gt[:, :, 2:3]
        mask = np.uint8(mask)
        return flow, mask

    @classmethod
    def readDispKITTI(cls, filename):
        disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
        valid = disp > 0.0
        flow = np.stack([-disp, np.zeros_like(disp)], -1)
        return flow, valid

    @classmethod
    def writeFlowKITTI(cls, filename, uv):
        uv = 64.0 * uv + 2**15
        valid = np.ones([uv.shape[0], uv.shape[1], 1])
        uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
        cv2.imwrite(filename, uv[..., ::-1])

    @classmethod
    def read_gen(cls, file_name, read_mask=False):
        ext = os.path.splitext(file_name)[-1]
        if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
            if read_mask:
                return imageio.imread(file_name)
            else:
                return Image.open(file_name)
        elif ext == '.bin' or ext == '.raw':
            return np.load(file_name)
        elif ext == '.flo':
            return cls.readFlow(file_name).astype(np.float32)
        elif ext == '.pfm':
            flow = cls.readPFM(file_name).astype(np.float32)
            if len(flow.shape) == 2:
                return flow
            else:
                return flow[:, :, :-1]
        else:
            raise ValueError('wrong file type: %s' % ext)


class Tmux_line():
    '''
    get error:  error connecting to /tmp/tmux-11502/default (No such file or directory)
    I meet this error when using this tmux tool in virtual environment (machine by rlaunch or zsh).
    The reason is that the directory /tmp/ is empty in the virtual environment.

    '''

    @classmethod
    def new_session(cls, session_name, first_window_name='first'):
        # '''  tmux new-session -s a -n editor -d
        # test:  new_session('a','b')
        # '''
        os.system("tmux new-session -s %s -n %s -d" %
                  (session_name, first_window_name))

    @classmethod
    def new_window(cls, session_name, window_name):
        # '''  tmux neww -a -n tool -t init
        # test:  new_session('a','b')  & new_window('a', 'c')
        # '''
        os.system("tmux neww -a -n %s -t %s" % (window_name, session_name))

    @classmethod
    def switch_window(cls, session_name, window_name):
        # ''' tmux attach -t [session_name]  这个暂时还是别用，会从python弹到tmux对应窗口里面的
        # test:  new_session('a','b')  & new_window('a', 'c') & new_window('a', 'd') & switch_window('a', 'b')
        # '''
        os.system("tmux attach -t %s:%s" % (session_name, window_name))

    @classmethod
    def split_window(cls, session_name, window_name, h_v='h', panel_number=0):
        # ''' tmux split-window -h -t development
        # h表示横着分, v表示竖着分
        # test:  new_session('a','b')  & new_window('a', 'c') & split_window('a', 'b', h_v='h', panel_number=0)
        # '''
        assert h_v in ['h', 'v']
        os.system("tmux split-window -%s -t %s:%s.%s" %
                  (h_v, session_name, window_name, panel_number))

    @classmethod
    def split_window_by_2(cls, session_name, window_name):
        # ''' 拆成4个panel '''
        cls.split_window(session_name, window_name, h_v='v',
                         panel_number=0)  # 拆称0和1，横着

    @classmethod
    def split_window_by_4(cls, session_name, window_name):
        # ''' 拆成4个panel '''
        # cls.split_window(session_name, window_name, h_v='h', panel_number=0)  # 拆称0和1，横着
        # cls.split_window(session_name, window_name, h_v='v', panel_number=0)
        # cls.split_window(session_name, window_name, h_v='v', panel_number=1)

        cls.split_window(session_name, window_name, h_v='h',
                         panel_number=0)  # 左右分两个
        cls.split_window(session_name, window_name, h_v='v', panel_number=1)
        cls.split_window(session_name, window_name, h_v='v', panel_number=0)

    @classmethod
    def split_window_by_8(cls, session_name, window_name):
        # ''' 先拆成4个panel '''
        cls.split_window_by_4(session_name, window_name)
        for i in range(4):
            cls.split_window(session_name,
                             window_name,
                             h_v='v',
                             panel_number=4 - 1 - i)

    @classmethod
    def split_window_by_16(cls, session_name, window_name):
        # ''' 先拆成8个panel '''
        cls.split_window_by_8(session_name, window_name)
        for i in range(8):
            cls.split_window(session_name,
                             window_name,
                             h_v='h',
                             panel_number=8 - 1 - i)

    @classmethod
    def run_command(cls,
                    session_name,
                    window_name,
                    panel_number=0,
                    command_line='ls'):
        com = "tmux send-keys -t %s:%s.%s '%s' C-m" % (
            session_name, window_name, panel_number, command_line)
        # print(com)
        os.system(com)

    @classmethod
    def demo(cls):
        # tmux kill-session -t a
        # demo()
        session_name = 'k'
        window_name = 'c'
        cls.new_session(session_name)
        cls.new_window(session_name, window_name)
        cls.split_window_by_16(session_name, window_name)
        for i in range(16):
            time.sleep(0.1)
            cls.run_command(session_name, window_name, i, command_line='ls')

    @classmethod
    def demo_run_commands(cls, demo_num=17):
        session_name = 's'
        line_ls = ['ls' for i in range(demo_num)]
        cls.run_task(task_ls=line_ls,
                     task_name='demo',
                     session_name=session_name)

    @classmethod
    def run_command_v2(cls,
                       session_name,
                       window_name,
                       panel_number=0,
                       command_line='ls',
                       **kwargs):
        for i in kwargs.keys():
            command_line += ' --%s %s' % (i, kwargs[i])
        cls.run_command(session_name,
                        window_name,
                        panel_number=panel_number,
                        command_line=command_line)

    @classmethod
    def run_task(cls, task_ls, task_name='demo', session_name='k'):
        # task_ls is a list that contains some string line. Each string line is a command line we want to run.
        N = len(task_ls)
        window_number = 0
        ind = -1

        def create_window(window_number_, panel_number=16):
            window_name = task_name + '_%s' % window_number_
            cls.new_window(session_name, window_name)
            # cls.new_window(session_name, window_name)
            if panel_number == 16:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_16(session_name, window_name)
            elif panel_number == 8:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_8(session_name, window_name)
            elif panel_number == 4:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_4(session_name, window_name)
            elif panel_number == 2:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_2(session_name, window_name)
            elif panel_number == 1:
                print('create a window with %s panels' % panel_number)
            else:
                pass
            window_number_ += 1
            return window_number_, window_name

        def run_16(data_ls, cnt, window_number_):
            for i in range(len(data_ls) // 16):
                # create window
                window_number_, window_name = create_window(window_number_,
                                                            panel_number=16)
                print(window_name)
                for j in range(16):
                    cnt += 1
                    if cnt >= N:
                        return cnt, window_number_
                    cls.run_command(session_name=session_name,
                                    window_name=window_name,
                                    panel_number=j,
                                    command_line=data_ls[cnt])
                    print(window_name, data_ls[cnt])
            return cnt, window_number_

        def run_one_window(data_ls, cnt, window_number_, panel_number):
            window_number_, window_name = create_window(
                window_number_, panel_number=panel_number)
            print(window_name)
            for i in range(panel_number):
                cnt += 1
                if cnt >= N:
                    return cnt, window_number_
                cls.run_command(session_name=session_name,
                                window_name=window_name,
                                panel_number=i,
                                command_line=data_ls[cnt])
                print(window_name, data_ls[cnt])
            return cnt, window_number_

        if N > 16:
            ind, window_number = run_16(task_ls,
                                        cnt=ind,
                                        window_number_=window_number)
        rest_number = N - ind - 1
        if rest_number > 8:
            ind, window_number = run_one_window(task_ls,
                                                cnt=ind,
                                                window_number_=window_number,
                                                panel_number=16)
        elif rest_number > 4:
            ind, window_number = run_one_window(task_ls,
                                                cnt=ind,
                                                window_number_=window_number,
                                                panel_number=8)
        elif rest_number > 2:
            ind, window_number = run_one_window(task_ls,
                                                cnt=ind,
                                                window_number_=window_number,
                                                panel_number=4)
        elif rest_number > 0:
            ind, window_number = run_one_window(task_ls,
                                                cnt=ind,
                                                window_number_=window_number,
                                                panel_number=2)
        else:
            pass
