"""Plot native matched probes without playback or level normalization.
Usage: python plot.py <weather-review-directory> <output-png>
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root, output = map(Path, sys.argv[1:])
def read(name, listener=0):
    p=root/(name+'-audio');r=json.loads((p/'run.json').read_text());rate,x=wavfile.read(p/f'listener-{listener}.wav')
    return rate,x[r['dspLatencySamples']:].astype(float)
def blocks(name, listener=0, duration=.25, peak=False):
    rate,x=read(name,listener);n=round(rate*duration);x=x[:len(x)//n*n].reshape(-1,n)
    y=np.max(abs(x),axis=1) if peak else 20*np.log10(np.maximum(1e-20,np.sqrt(np.mean(x*x,axis=1))))
    return (np.arange(len(x))+.5)*n/rate,y
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
for name,label,color in [('wind-baseline','Original wind','#ab5059'),('wind-dense','Denser wind','#16827b')]:
    t,y=blocks(name);axes[0].plot(t,y,label=label,color=color,lw=1.6)
axes[0].set_title('Wind outside: coverage improves at unchanged source gain');axes[0].set_ylabel('RMS dBFS');axes[0].set_ylim(-125,-25)
for name,label,color in [('rain-outside','Outside','#b17c17'),('rain-inside','Closed interior','#435f9b')]:
    t,y=blocks(name);axes[1].plot(t,y,label=label,color=color,lw=1.6)
axes[1].set_title('Roof + ground rain: indoor transients remain');axes[1].set_ylabel('RMS dBFS');axes[1].set_ylim(-125,-25)
for name,label,color in [('rain-inside','Transport seed 0','#435f9b'),('rain-inside-seed1','Transport seed 1','#b35f93')]:
    t,y=blocks(name,duration=.01,peak=True);axes[2].plot(t,y,label=label,color=color,lw=1)
axes[2].set_title('Identical rain events and geometry; only the transport seed changes');axes[2].set_ylabel('10 ms absolute peak');axes[2].set_xlabel('Seconds after emission starts')
for ax in axes:
    ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=8);ax.set_xlim(0,7.5);ax.axvline(6,color='#555555',ls=':',lw=1)
axes[2].text(6.04,.98,'Emission stops',transform=axes[2].get_xaxis_transform(),va='top',fontsize=8)
fig.savefig(output,dpi=160)
