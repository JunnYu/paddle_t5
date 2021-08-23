# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import json
import os

from . import DatasetBuilder

__all__ = ['CnnDailyMail']


class CnnDailyMail(DatasetBuilder):
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5', 'URL'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('cnn_dailymail_train.json'),
            'd50a105f9689e72be7d79adbba0ae224',
            'https://paddlenlp.bj.bcebos.com/datasets/cnn_dailymail_train.json'
        ),
        'dev': META_INFO(
            os.path.join('cnn_dailymail_dev.json'),
            'e36a295c1cb8c6b9fb28015907a42d9e',
            'https://paddlenlp.bj.bcebos.com/cnn_dailymail_dev.json'
        ),
        'test': META_INFO(
            os.path.join('cnn_dailymail_test.json'),
            '91a6cf060e1283f05fcc6a2027238379',
            'https://paddlenlp.bj.bcebos.com/datasets/cnn_dailymail_test.json'
        )
    }

    def _get_data(self, mode, **kwargs):
        # default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash, URL = self.SPLITS[mode]
        # 本地文件夹
        default_root = "caches/cnndailymail"
        fullname = os.path.join(default_root, filename)

        return fullname

    def _read(self, filename, *args):
        with open(filename, "r", encoding="utf8") as f:
            for line in f.readlines():
                dic = json.loads(line)
                yield {
                    "article": dic["article"],
                    "highlights": dic["highlights"],
                }
