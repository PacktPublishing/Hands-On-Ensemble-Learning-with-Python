# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:19:55 2019

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

base_errors = (test_meta_data.transpose() - test_y).transpose()
prediction_errors = ensemble_predictions - test_y

for i in range(len(prediction_errors)):
    if not prediction_errors[i] == 0.0:
        print(base_errors[i,:])