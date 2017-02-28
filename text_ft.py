# coding=utf-8

'''
@author:lruoran

created at 17-2-28 下午1:27
'''

import fasttext

clf = fasttext.supervised(input_file='data/rt-polaritydata/rt_ft.train', output='data/rt-polaritydata/ft_model',
                          label_prefix='__label__', epoch=10, silent=0, dim=300, thread=12, neg=5, ws=5)

result = clf.test(r"data/rt-polaritydata/rt_ft.test")
print 'P@1:', result.precision
print 'R@1:', result.recall
print 'Number of examples:', result.nexamples
