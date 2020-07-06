# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri yukleme
veriler = pd.read_csv('veriler.csv')

#veri on isleme 
boy = veriler[['boy']]

boykilo = veriler[['boy','kilo']]
