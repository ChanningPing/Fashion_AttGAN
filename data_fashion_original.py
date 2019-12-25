# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tflib.utils import session
import argparse
import sys
from importlib import reload
reload(sys)
#sys.setdefaultencoding('utf8')




def batch_dataset(dataset, batch_size, prefetch_batch=2, drop_remainder=True, filter=None,
                  map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1):
    if filter:
        dataset = dataset.filter(filter)

    if map_func:
        dataset = dataset.map(map_func, num_parallel_calls=num_threads)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    if drop_remainder:
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    else:
        dataset = dataset.batch(batch_size)

    dataset = dataset.repeat(repeat).prefetch(prefetch_batch)

    return dataset


def disk_image_batch_dataset(img_paths, batch_size, labels=None, prefetch_batch=2, drop_remainder=True, filter=None,
                             map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1):
    """Disk image batch dataset.

    This function is suitable for jpg and png files

    img_paths: string list or 1-D tensor, each of which is an iamge path
    labels: label list/tuple_of_list or tensor/tuple_of_tensor, each of which is a corresponding label
    """
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    elif isinstance(labels, tuple):
        dataset = tf.data.Dataset.from_tensor_slices((img_paths,) + tuple(labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    def parse_func(path, *label):
        img = tf.read_file(path)
        img = tf.image.decode_png(img, 3)
        return (img,) + label

    if map_func:
        def map_func_(*args):
            return map_func(*parse_func(*args))
    else:
        map_func_ = parse_func

    # dataset = dataset.map(parse_func, num_parallel_calls=num_threads) is slower

    dataset = batch_dataset(dataset, batch_size, prefetch_batch, drop_remainder, filter,
                            map_func_, num_threads, shuffle, buffer_size, repeat)

    return dataset


class Dataset(object):

    def __init__(self):
        self._dataset = None
        self._iterator = None
        self._batch_op = None
        self._sess = None

        self._is_eager = tf.executing_eagerly()
        self._eager_iterator = None

    def __del__(self):
        if self._sess:
            self._sess.close()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            b = self.get_next()
        except:
            raise StopIteration
        else:
            return b

    next = __next__

    def get_next(self):
        if self._is_eager:
            return self._eager_iterator.get_next()
        else:
            return self._sess.run(self._batch_op)

    def reset(self, feed_dict={}):
        if self._is_eager:
            self._eager_iterator = tfe.Iterator(self._dataset)
        else:
            self._sess.run(self._iterator.initializer, feed_dict=feed_dict)

    def _bulid(self, dataset, sess=None):
        self._dataset = dataset

        if self._is_eager:
            self._eager_iterator = tfe.Iterator(dataset)
        else:
            self._iterator = dataset.make_initializable_iterator()
            self._batch_op = self._iterator.get_next()
            if sess:
                self._sess = sess
            else:
                self._sess = session()

        try:
            self.reset()
        except:
            pass

    @property
    def dataset(self):
        return self._dataset

    @property
    def iterator(self):
        return self._iterator

    @property
    def batch_op(self):
        return self._batch_op


class Celeba(Dataset):
    att_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
                'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
                'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
                'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
                'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
                'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
                'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
                'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
                'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
                'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
                'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
                'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
                'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

    att_dict = {
        "连衣裙": 0, "女式背心": 1, "女式马夹": 2, "女式针织衫": 3, "女式毛衣": 4, "女式POLO衫": 5, "女式卫衣": 6, "女式衬衫": 7, "女式T恤": 8, "女式皮草": 9,
        "女式皮衣": 10, "女式羽绒": 11, "女式大衣": 12, "女式风衣/长外套": 13, "女式夹克": 14, "女式西服": 15, "女式短外套": 16, "其他": 17, "泳衣泳裤": 18,
        "连体裤": 19, "卫衣": 20, "冲锋衣／抓绒衣": 21, "家居服/睡衣": 22, "其他": 23, "素色": 24, "拼色": 25, "渐变色": 26, "混杂色": 27,
        "细横条纹": 28, "粗横条纹": 29, "竖条纹": 30, "暗纹": 31, "小方格": 32, "大方格": 33, "菱形格": 34, "千鸟格 ": 35, "波点": 36, "个性印花": 37,
        "图案": 38, "印花": 39, "几何印花": 40, "豹纹": 41, "虎纹": 42, "斑马纹": 43, "蛇纹": 44, "迷彩": 45, "提花": 46, "其他": 47,
        "大理石纹": 48, "次花色模式": 49, "躯干素色": 50, "局部素色": 51, "拼色": 52, "渐变色": 53, "混杂色": 54, "细横条纹": 55, "粗横条纹": 56,
        "竖条纹": 57, "暗纹": 58, "小方格": 59, "大方格": 60, "菱形格": 61, "千鸟格 ": 62, "波点": 63, "个性印花": 64, "图案": 65, "印花": 66,
        "几何印花": 67, "豹纹": 68, "虎纹": 69, "斑马纹": 70, "蛇纹": 71, "迷彩": 72, "提花": 73, "其他": 74, "大理石纹": 75, "组合模式": 76,
        "拼接": 77, "叠加": 78, "图案": 79, "方块印图": 80, "字母数字": 81, "字母数字+其他": 82, "植物花卉": 83, " 人脸人像": 84, "动物卡通": 85,
        "骷髅头": 86, "建筑风景": 87, "其他图案": 88, "无纹理": 89, "全蕾丝": 90, "全镂空": 91, " 全网纱/透视": 92, "蕾丝拼接": 93, "镂空拼接": 94,
        "网纱/拼接": 95, "铰花": 96, "绒毛": 97, "压线": 98, "压线拼接": 99, "多层花边": 100, "褶皱": 101, "百褶": 102, "其他纹理": 103,
        "其他纹理拼接": 104, "翻领": 105, "高领": 106, "立领": 107, "毛领": 108, "西装领（平驳领/创驳领）": 109, "翻领(西装)": 110, "娃娃领/公主领": 111,
        "荡领": 112, "衬衫领/衬衣领": 113, "POLO领/T恤领": 114, "连帽领": 115, "棒球领": 116, "围脖领/堆领/堆堆领": 117, "系带领": 118, "青果领": 119,
        "圆领": 120, "V领": 121, "小V领": 122, "方领": 123, "U领": 124, "一字领": 125, "吊带领": 126, "其他无领": 127, "露肩一字领/抹胸": 128,
        "斜肩领": 129, "其他有领": 130, "无袖": 131, "短袖": 132, "盖肩袖": 133, "中袖/5分袖": 134, "6分袖/7分袖/8分袖": 135, "9分袖": 136,
        "长袖": 137, "腰部": 138, "髋部": 139, "裆部(超短裙)": 140, "大腿(短裙)": 141, "膝盖(中裙)": 142, "小腿(中长裙)": 143, "脚踝（及踝裙）": 144,
        "脚(及地裙)": 145, "前短后长": 146, "其它不规则": 147, "宽松": 148, "标准": 149, "修身": 150, "紧身": 151, "套头": 152, "单排扣": 153,
        "双排扣": 154, "居中拉链": 155, "斜拉链": 156, "暗门襟": 157, "开襟": 158, "一粒扣/暗扣": 159, "牛角扣/长条扣": 160, "裙型": 161,
        "鱼尾裙": 162, "蛋糕裙": 163, "节裙/塔裙": 164, "百褶裙": 165, "泡泡裙": 166, "花苞裙": 167, "蓬蓬裙": 168, "A字裙": 169, "伞裙／喇叭裙": 170,
        "包臀裙／一步裙": 171, "直筒裙": 172, "其他": 173, "其他": 174, "棉/麻/棉麻": 175, "雪纺": 176, "丝绸": 177, "牛仔": 178, "毛呢": 179,
        "毛线": 180, "皮/pu": 181, "皮草": 182, "其他": 183, "其他": 184, "红色": 185, "粉色": 186, "橙色": 187, "黄色": 188, "绿色": 189,
        "蓝色": 190, "紫色": 191, "灰色": 192, "黑色": 193, "白色": 194, "米色": 195, "棕色": 196, "褐色": 197, "咖色": 198, "驼色": 199,
        "杏色": 200, "青色": 201, "藏青色": 202, "银色": 203, "花色": 204, "其他": 205, "金色": 206, "次色": 207, "红色": 208, "粉色": 209,
        "橙色": 210, "黄色": 211, "绿色": 212, "蓝色": 213, "紫色": 214, "灰色": 215, "黑色": 216, "白色": 217, "米色": 218, "棕色": 219,
        "褐色": 220, "咖色": 221, "驼色": 222, "杏色": 223, "青色": 224, "藏青色": 225, "银色": 226, "花色": 227, "其他": 228, "金色": 229,
        "主元素": 230, "logo": 231, "刺绣": 232, "口袋": 233, "破洞": 234, "亮片/金属箔": 235, "铆钉/钉珠/水钻": 236, "流苏": 237, "蕾丝": 238,
        "镂空": 239, "补丁/贴布": 240, "系带": 241, "饰边": 242, "其他元素": 243, "镂空元素": 244, "蕾丝元素": 245, "次元素": 246, "logo": 247,
        "刺绣": 248, "口袋": 249, "破洞": 250, "亮片/金属箔": 251, "铆钉/钉珠/水钻": 252, "流苏": 253, "蕾丝": 254, "镂空": 255, "补丁/贴布": 256,
        "系带": 257, "饰边": 258, "其他元素": 259, "镂空元素": 260, "蕾丝元素": 261, "男童": 262, "男士": 263, "中性成人": 264, "女士": 265,
        "中性儿童": 266, "女童": 267
    }

    att_dict = {
        "连衣裙": 0, "女式背心": 1, "女式马夹": 2, "女式针织衫": 3, "女式毛衣": 4, "女式POLO衫": 5, "女式卫衣": 6, "女式衬衫": 7, "女式T恤": 8, "女式皮草": 9,
        "女式皮衣": 10, "女式羽绒": 11, "女式大衣": 12, "女式风衣/长外套": 13, "女式夹克": 14, "女式西服": 15, "女式短外套": 16, "其他": 17, "泳衣泳裤": 18,
        "连体裤": 19, "卫衣": 20, "冲锋衣／抓绒衣": 21, "家居服/睡衣": 22, "其他": 23, "素色": 24, "拼色": 25, "渐变色": 26, "混杂色": 27,
        "细横条纹": 28, "粗横条纹": 29, "竖条纹": 30, "暗纹": 31, "小方格": 32, "大方格": 33, "菱形格": 34, "千鸟格 ": 35, "波点": 36, "个性印花": 37,
        "图案": 38, "印花": 39, "几何印花": 40, "豹纹": 41, "虎纹": 42, "斑马纹": 43, "蛇纹": 44, "迷彩": 45, "提花": 46, "其他": 47,
        "大理石纹": 48}

    att_dict = {
        "无袖": 131, "短袖": 132, "盖肩袖": 133, "中袖/5分袖": 134, "6分袖/7分袖/8分袖": 135, "9分袖": 136, "长袖": 137,
        "红色": 185, "粉色": 186, "橙色": 187, "黄色": 188, "绿色": 189,
        "蓝色": 190, "紫色": 191, "灰色": 192, "黑色": 193, "白色": 194, "米色": 195, "棕色": 196, "褐色": 197, "咖色": 198, "驼色": 199,
        "杏色": 200, "青色": 201, "藏青色": 202, "银色": 203, "花色": 204,  "金色": 206
    }






    def __init__(self, data_dir, atts, img_resize, batch_size, prefetch_batch=2, drop_remainder=True,
                 num_threads=16, shuffle=True, buffer_size=4096, repeat=-1, sess=None, part='train', crop=True):
        super(Celeba, self).__init__()

        list_file = os.path.join(data_dir, 'list_attr_upperbody.txt')
        if crop:
            img_dir_jpg = os.path.join(data_dir, '')
            img_dir_png = os.path.join(data_dir, 'png')
        else:
            img_dir_jpg = os.path.join(data_dir, '')
            img_dir_png = os.path.join(data_dir, 'png')

        names = np.loadtxt(list_file, skiprows=2, usecols=[0], dtype=np.str)
        if os.path.exists(img_dir_png):
            img_paths = [os.path.join(img_dir_png, name.replace('jpg', 'png')) for name in names]
        elif os.path.exists(img_dir_jpg):
            img_paths = [os.path.join(img_dir_jpg, name) for name in names]
        else:  # not sure why add this
            img_paths = [os.path.join(img_dir_jpg, name) for name in names]

        att_id = [Celeba.att_dict[att] + 1 for att in atts]

        labels = np.loadtxt(list_file, skiprows=2, usecols=att_id, dtype=np.int64)

        if img_resize == 64:
            # crop as how VAE/GAN do
            offset_h = 40
            offset_w = 15
            img_size = 148
        else:
            offset_h = 26
            offset_w = 3
            img_size = 170

        def _map_func(img, label):
            if crop:
                img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, img_size, img_size)
            #img = tf.image.resize_images(img, [img_size, img_size]) / 127.5 - 1

            #img = tf.cast(img, tf.float32)
            #img = tf.image.resize_images(img, [img_resize, img_resize]) / 127.5 - 1

            # or
            print('crop value')
            print(crop)
            img = tf.image.resize_images(img, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            label = (label + 1) // 2
            return img, label

        if part == 'test':
            drop_remainder = False
            shuffle = False
            repeat = 1
            img_paths = img_paths[14000:]
            labels = labels[14000:]
        elif part == 'val':
            img_paths = img_paths[14000:]
            labels = labels[14000:]
        else:
            img_paths = img_paths[:14000]
            labels = labels[:14000]

        dataset = disk_image_batch_dataset(img_paths=img_paths,
                                           labels=labels,
                                           batch_size=batch_size,
                                           prefetch_batch=prefetch_batch,
                                           drop_remainder=drop_remainder,
                                           map_func=_map_func,
                                           num_threads=num_threads,
                                           shuffle=shuffle,
                                           buffer_size=buffer_size,
                                           repeat=repeat)
        self._bulid(dataset, sess)

        self._img_num = len(img_paths)

    def __len__(self):
        return self._img_num

    @staticmethod
    def check_attribute_conflict(att_batch, att_name, att_names):
        def _set(att, value, att_name):
            if att_name in att_names:
                att[att_names.index(att_name)] = value

        att_id = att_names.index(att_name)

        for att in att_batch:
            if att_name in ["无袖","短袖","盖肩袖","中袖/5分袖","6分袖/7分袖/8分袖","9分袖","长袖"] \
                    and att[att_id] == 1:
                for n in ["无袖","短袖","盖肩袖","中袖/5分袖","6分袖/7分袖/8分袖","9分袖","长袖"]:
                    if n != att_name:
                        _set(att, 0, n)
            elif att_name in ["红色","粉色","橙色","黄色","绿色","蓝色","紫色","灰色","黑色","白色","米色","棕色","褐色","咖色","驼色","杏色","青色","藏青色","银色","花色","金色"] \
                    and att[att_id] == 1:
                for n in ["红色","粉色","橙色","黄色","绿色","蓝色","紫色","灰色","黑色","白色","米色","棕色","褐色","咖色","驼色","杏色","青色","藏青色","银色","花色","金色"]:
                    if n != att_name:
                        _set(att, 0, n)



        return att_batch

    @staticmethod
    def change_attribute(att_batch, max_change_num):
        import random
        from random import randint

        for att in att_batch:

            #print('att before change')
            #print(att)
            attr_intervals = [[0,23],[23,49],[49,76],[76,79],[79,89],[89,105],[105,131],[131,138],[138,148],[148,152],[152,161],[161,174],[174,184],[184,207],[207,230],[230,246],[246,262],[262,268]]
            attr_intervals = [[0, 23], [23, 49]]
            intervals_to_change = random.sample(attr_intervals, max_change_num)
            for intervals in intervals_to_change:
                for i in range(intervals[0],intervals[1]):
                    att[i] = 0
                change_idx = randint(intervals[0],intervals[1]-1)
                att[change_idx] = 1

            #print('att after change')
            #print(att)
        return att_batch




if __name__ == '__main__':
    import imlib as im

    atts = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male',
            'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
    atts = ["连衣裙","女式背心","女式马夹","女式针织衫","女式毛衣","女式POLO衫","女式卫衣","女式衬衫","女式T恤","女式皮草","女式皮衣","女式羽绒","女式大衣","女式风衣/长外套","女式夹克","女式西服","女式短外套","其他","泳衣泳裤","连体裤","卫衣","冲锋衣／抓绒衣","家居服/睡衣","其他","素色","拼色","渐变色","混杂色","细横条纹","粗横条纹","竖条纹","暗纹","小方格","大方格","菱形格","千鸟格 ","波点","个性印花","图案","印花","几何印花","豹纹","虎纹","斑马纹","蛇纹","迷彩","提花","其他","大理石纹","次花色模式","躯干素色","局部素色","拼色","渐变色","混杂色","细横条纹","粗横条纹","竖条纹","暗纹","小方格","大方格","菱形格","千鸟格 ","波点","个性印花","图案","印花","几何印花","豹纹","虎纹","斑马纹","蛇纹","迷彩","提花","其他","大理石纹","组合模式","拼接","叠加","图案","方块印图","字母数字","字母数字+其他","植物花卉"," 人脸人像","动物卡通","骷髅头","建筑风景","其他图案","无纹理","全蕾丝","全镂空"," 全网纱/透视","蕾丝拼接","镂空拼接","网纱/拼接","铰花","绒毛","压线","压线拼接","多层花边","褶皱","百褶","其他纹理","其他纹理拼接","翻领","高领","立领","毛领","西装领（平驳领/创驳领）","翻领(西装)","娃娃领/公主领","荡领","衬衫领/衬衣领","POLO领/T恤领","连帽领","棒球领","围脖领/堆领/堆堆领","系带领","青果领","圆领","V领","小V领","方领","U领","一字领","吊带领","其他无领","露肩一字领/抹胸","斜肩领","其他有领","无袖","短袖","盖肩袖","中袖/5分袖","6分袖/7分袖/8分袖","9分袖","长袖","腰部","髋部","裆部(超短裙)","大腿(短裙)","膝盖(中裙)","小腿(中长裙)","脚踝（及踝裙）","脚(及地裙)","前短后长","其它不规则","宽松","标准","修身","紧身","套头","单排扣","双排扣","居中拉链","斜拉链","暗门襟","开襟","一粒扣/暗扣","牛角扣/长条扣","裙型","鱼尾裙","蛋糕裙","节裙/塔裙","百褶裙","泡泡裙","花苞裙","蓬蓬裙","A字裙","伞裙／喇叭裙","包臀裙／一步裙","直筒裙","其他","其他","棉/麻/棉麻","雪纺","丝绸","牛仔","毛呢","毛线","皮/pu","皮草","其他","其他","红色","粉色","橙色","黄色","绿色","蓝色","紫色","灰色","黑色","白色","米色","棕色","褐色","咖色","驼色","杏色","青色","藏青色","银色","花色","其他","金色","次色","红色","粉色","橙色","黄色","绿色","蓝色","紫色","灰色","黑色","白色","米色","棕色","褐色","咖色","驼色","杏色","青色","藏青色","银色","花色","其他","金色","主元素","logo","刺绣","口袋","破洞","亮片/金属箔","铆钉/钉珠/水钻","流苏","蕾丝","镂空","补丁/贴布","系带","饰边","其他元素","镂空元素","蕾丝元素","次元素","logo","刺绣","口袋","破洞","亮片/金属箔","铆钉/钉珠/水钻","流苏","蕾丝","镂空","补丁/贴布","系带","饰边","其他元素","镂空元素","蕾丝元素","男童","男士","中性成人","女士","中性儿童","女童"]
    atts = ["无袖","短袖","盖肩袖","中袖/5分袖","6分袖/7分袖/8分袖","9分袖","长袖","红色","粉色","橙色","黄色","绿色","蓝色","紫色","灰色","黑色","白色","米色","棕色","褐色","咖色","驼色","杏色","青色","藏青色","银色","花色","金色"]

    data = Celeba('./data', atts, 128, 32, part='val')
    batch = data.get_next()
    print(len(data))
    print(batch[1][1], batch[1].dtype)
    print(batch[0].min(), batch[1].max(), batch[0].dtype)
    im.imshow(batch[0][1])
    im.show()
