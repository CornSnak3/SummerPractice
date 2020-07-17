import argparse
import logging
import matplotlib
import copy
import os
import numpy as np
import glob
import pyAudioAnalysis
import fileinput
import scipy.io.wavfile as wavfile
import matplotlib.patches
import pandas as pd
from tkinter import *
from tkinter.filedialog import *
from pyAudioAnalysis import ShortTermFeatures as sF
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioVisualization as aV
from pyAudioAnalysis import audioBasicIO
from matplotlib import pyplot as plot
from s4d.utils import *
from s4d.diar import Diar
from s4d import viterbi, segmentation
from s4d.clustering import hac_bic
from sidekit.sidekit_io import init_logging
from s4d.gui.dendrogram import plot_dendrogram

def _open():
    op = askopenfilename(filetypes=[("WAVE sound","*.wav")])
    labelFileNameSound.config(state=NORMAL)
    labelFileNameSound.delete(0, END)
    labelFileNameSound.insert(0, op)
    labelFileNameSound.config(state=DISABLED)
    op = askopenfilename(filetypes=[(".segments", "*.segments")])
    labelFileNameSegment.config(state=NORMAL)
    labelFileNameSegment.delete(0, END)
    labelFileNameSegment.insert(0, op)
    labelFileNameSegment.config(state=DISABLED)

def pyAudioDiar():
    duration, result = aS.speaker_diarization(labelFileNameSound.get(), int(labelNumberOfSpeakers.get()), lda_dim=0, plot_res=False);
    show = 'diarizationExample'
    input_show = labelFileNameSound.get()
    input_sad = None
    win_size = 250
    thr_l = 2
    thr_h = 3
    thr_vit = -250
    wdir = os.path.join('out', show)
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    fs = get_feature_server(input_show, feature_server_type='basic')
    cep, _ = fs.load(show)
    cep.shape

    if input_sad is not None:
        init_diar = Diar.read_seg(input_sad)
        init_diar.pack(50)
    else:
        init_diar = segmentation.init_seg(cep, show)

    seg_diar = segmentation.segmentation(cep, init_diar, win_size)

    bicl_diar = segmentation.bic_linear(cep, seg_diar, thr_l, sr=False)

    bic = hac_bic.HAC_BIC(cep, bicl_diar, thr_h, sr=False)
    bich_diar = bic.perform(to_the_end=True)

    vit_diar = viterbi.viterbi_decoding(cep, bich_diar, thr_vit)
    resList = []
    currentPosition = 0
    for row in vit_diar:
        speakerValue = int(row[1][1:])
        while currentPosition < (row[3] + row[4]):
            resList.append(speakerValue)
            currentPosition += 20

    currentPosition = 0
    realityList = []
    realityFile = pd.read_csv(labelFileNameSegment.get(), delimiter='\t',
                              encoding='utf-8', names=['start', 'end', 'speaker'])
    for index, row in realityFile.iterrows():
        speakerValue = int(row['speaker'][1:])
        while currentPosition < row['end']:
            realityList.append(int(speakerValue))
            currentPosition += 0.2



    plot.subplot(3, 1, 2)
    plot.title("s4d:")
    plot.plot(np.arange(0, duration, duration / len(resList)), resList, 'ro')
    plot.subplot(3, 1, 1)
    plot.title("Реальность:")
    plot.plot(np.arange(0, duration, duration / len(realityList)), realityList, 'bo')
    plot.subplot(3, 1, 3)
    plot.title("pyPlotAudio:")
    plot.plot(np.arange(0, duration, duration / len(result)), result, 'go')
    plot.show()


matplotlib.use('TkAgg')

window = Tk()
window.title("Speech diarization")
labelFileNameSound = Entry(window, width=50);
labelFileNameSound.grid(column=0, row=0);
labelFileNameSound.config(state=DISABLED)
labelNumberOfSpeakers = Entry(window, width=4)
labelNumberOfSpeakers.grid(column=1, row=0)
labelFileNameSegment = Entry(window, width=50);
labelFileNameSegment.grid(column=0, row=1);
labelFileNameSegment.config(state=DISABLED)
btnOpen = Button(text="Выбрать файл", command=_open)
btnOpen.grid(column=1, row=1)
btnDiar = Button(text="Начать!", command=pyAudioDiar)
btnDiar.grid(column=1, row=2)
mainloop()