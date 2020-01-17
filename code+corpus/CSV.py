#!/usr/bin/env python
# coding: utf-8

import numpy as np

def predictCSV(model,test_y):
    result = model.predict(test_y)
    np.savetxt("prediction.csv", np.dstack((np.arange(0, result.size),result))[0],"%s,%s",header="Id,Category")
