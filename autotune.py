"""
autotune.py - module generate test data for autotune

Usage:
	import autotune
	autotune_baseline_test()
	autotune.rms_spectrum_test(song='tainted', tuning_f0=110.) # RMS power of equal-temperament bins for song 'tainted'
	autotune.mixture_test(song='tainted', tuning_f0=110.) # apply autotune algorithm to song/vocals.wav and mix with song/background.wav
	autotune.predominant_melody_test(song='tained', tuning_f0=110.)

Author: Michael A. Casey
Copyright (C) 2015, Bregman Media Labs, Dartmouth College
License: Apache 2.0, see LICENSE file
""" 

from bregman.suite import * # the Bregman audio processing toolkit
import matplotlib
from matplotlib.pyplot import *
from matplotlib.mlab import rms_flat
from numpy import *
import os, glob
import scikits.audiolab as audio

try:
	import voweltimbre as sung
except:
	print "warning: voweltimbre not installed"
try:
	from essentia.standard import *
except:
	print "warning: essentia not installed"
	
equal_temperament = array(TuningSystem().equal_temperament()) # Ratios for equal temperament tuning
just_intonation = array(TuningSystem().just_intonation()) # Ratios for equal temperament tuning
pythagorean = array(TuningSystem().Pythagorean()) # Ratios for equal temperament tuning

major_scale = [0,2,4,5,7,9,11,12]
f0 = 440 # tuning reference frequency

def gen_audio(f0=440, tuning=equal_temperament, scale=major_scale, filename=None):
	# Audio envelope for a note
	env = r_[linspace(0,1,440),linspace(1,.8,220), .8*ones(22050-3*660), linspace(.8,0,660), zeros(660)]

	# Construct the scale at the given tuning
	x = hstack([env * harmonics(f0=f, num_harmonics=10, num_points=22050) for f in f0*tuning[scale]])

	if filename is None:
		play(balance_signal(x))
	else:
		wavwrite(x,filename,44100)

def gen_test_signals():
	gen_audio(f0, equal_temperament, major_scale, 'A440_Equal_Major.wav')
	gen_audio(f0, just_intonation, major_scale, 'A440_Just_Major.wav')
	gen_audio(f0, pythagorean, major_scale, 'A440_Pythagorean_Major.wav')

	# Detune entire scale
	f1 = f0*2**(0.4/12.) # A440 * {0.4 semitone}
	gen_audio(f1, equal_temperament, major_scale, 'A450_Equal_Major.wav')
	gen_audio(f1, just_intonation, major_scale, 'A450_Just_Major.wav')
	gen_audio(f1, pythagorean, major_scale, 'A450_Pythagorean_Major.wav')

def load_signals(dir_expr="*.wav"):
	flist = sorted(glob.glob(dir_expr))
	if len(flist) ==0 :
		raise ValueError("No files found.")
	X = {}
	ext = flist[0].split('.')[-1][:3]
	if ext=='wav':
		afun = wavread
	elif ext=='aif':
		afun = aiffread
	else:
		raise ValueError("Unrecognized audio file extension: %s"%ext)
	for f in flist:
		x, sr, fmt = afun(f)
		f = f.split(os.sep)[-1] if len(f.split(os.sep)) else f
		f = f.split('.')[0].replace(' ','_')
		X[f] = x
	return X

def peaks_to_autotuned_spectrum(audio, peaks, factor=1.0, N=8192, H=2048, SR=44100.):
	"""
	Peaks to autotuned short-time Fourier transform
	inputs:
		audio - float array
		peaks - peak dict of 'freqs' and 'mags' per time-point
		factor - amount to autotune [1.0=100%]
		N     - fft length
		H     - fft hop
		SR    - audio sample rate
	outputs:
		bregman.LogFrequencySpectrum
	"""
	freqs, mags = peaks['freqs'], peaks['mags']
	F = features.LinearFrequencySpectrum(audio, nfft=N, wfft=N, nhop=H)
	eq_freqs = 55*2**(arange(0,8.5,1/12.))
	eq_bins = [argmin(abs(F._fftfrqs-f)) for f in eq_freqs]
	Xhat = zeros(F.X.shape)	
	T = Xhat.shape[1]
	for t in xrange(len(freqs)):
		if t<T:
			for i,(f,a) in enumerate(zip(freqs[t],mags[t])):
				if i==0:
					eq_freq = eq_freqs[argmin(abs(eq_freqs-f))]
					eq_ratio = eq_freq / f # fundamental frequency
				df = f * (eq_ratio - 1.0)
				f_idx = argmin(abs(F._fftfrqs - (f + factor * df))) # harmonics
				Xhat[f_idx,t]=a
	F.X = Xhat
	return F

def auto_tune(fname, factor=1.0):
	X = load_signals(fname)
	x = array(X[X.keys()[0]][:,0],dtype='f')
	peaks = sung.predominant_harmonics(x, fname)
	stft = peaks_to_autotuned_spectrum(x, peaks, factor)
	xhat = stft.inverse(stft.X)	
	return xhat

def normalize(A, axis=None):
    Ashape = A.shape
    try:
        norm = A.sum(axis) + EPS
    except TypeError:
        norm = A.copy()
        for ax in reversed(sorted(axis)):
            norm = norm.sum(ax)
        norm += EPS
    if axis:
        nshape = np.array(Ashape)
        nshape[axis] = 1
        norm.shape = nshape
    return A / norm

def baseline_test():
	# load reference, detuned, and auto-tuned signals
	xdata = load_signals("A*_Equal_Major.wav")
	ydata = load_signals("A450_Equal_Major_100.wav")
	x0 = xdata["A440_Equal_Major"] # reference
	x1 = xdata['A450_Equal_Major'] # detuned (f0)
	y1 = ydata['A450_Equal_Major_100'] # autotuned (processed signal)
	# Spectral bin resolution = 2.0Hz, time resolution = 2.0Hz
	n = 44100 / 2
	X0 = LinearFrequencySpectrum(x0, nfft=n, wfft=n, nhop=n)
	X1 = LinearFrequencySpectrum(x1, nfft=n, wfft=n, nhop=n)
	Y1 = LinearFrequencySpectrum(y1, nfft=n, wfft=n, nhop=n)
	freqs = X0._fftfrqs
	figure()
	semilogx(freqs,X1.X[:,0])
	semilogx(freqs,Y1.X[:,0])
	semilogx(freqs,X0.X[:,0],'--')
	title('Melodyne shifts harmonics of A450Hz to A440Hz', fontsize=20)
	xlabel('Frequency (Hz)',fontsize=20)
	ylabel('Power',fontsize=20)
	legend(['450Hz Original','Autotune','440Hz Reference'],loc=0)
	eq_freqs = 110*2**(arange(0,6,1/12.))
	eq_bins = [argmin(abs(X0._fftfrqs-f)) for f in eq_freqs]
	ax = axis()
	plot(c_[X0._fftfrqs[eq_bins],X0._fftfrqs[eq_bins]].T,c_[[ax[2]]*len(eq_freqs),[ax[3]]*len(eq_freqs)].T,'k--')

def mixture_test(song='tainted'):
	"""
	Display spectral profiles of original and autotuned mixture spectra
	inputs:
		song - directory name of song (contains: song vocals.wav and background.wav)
	outputs:
		mix_000, mix_100 - mixed vocals and background for nontuned and autotuned vocals
	"""
	X = load_signals(song+os.sep+'*.wav')
	x0 = X['vocals']
	x1 = X['background']
	xhat0 = auto_tune(song+os.sep+'vocals.wav',0.0)[:len(x0)] # no autotune
	xhat1 = auto_tune(song+os.sep+'vocals.wav',1.0)[:len(x0)] # autotuned to 440Hz
	mix0 = (balance_signal(c_[xhat0,xhat0])+balance_signal(x1))/2.0 # background+vocals no autotune
	mix1 = (balance_signal(c_[xhat1,xhat1])+balance_signal(x1))/2.0 # background+vocals with autotune
	# Short-time Fourier analysis
	F0 = LinearFrequencySpectrum(mix0,nfft=8192,wfft=8192,nhop=2048)
	F1 = LinearFrequencySpectrum(mix1,nfft=8192,wfft=8192,nhop=2048)
	eq_freqs = 110*2**(arange(0,5,1/12.))
	eq_bins = [argmin(abs(F1._fftfrqs-f)) for f in eq_freqs]
	# Plot spectra and ideal autotuned pitch bins
	figure()
	semilogx(F0._fftfrqs, normalize(F0.X).mean(1))
	semilogx(F1._fftfrqs, normalize(F1.X).mean(1))
	ax = axis()
	plot(c_[F0._fftfrqs[eq_bins],F0._fftfrqs[eq_bins]].T,c_[[ax[2]]*len(eq_freqs),[ax[3]]*len(eq_freqs)].T,'k--')
	legend(['Original vocals','Autotuned vocals','ET pitch'],loc=0)
	title(song+': untuned/tuned vocals mixed with background', fontsize=20)
	xlabel('Frequency (Hz)',fontsize=20)
	ylabel('Power',fontsize=20)
	# Calculate RMS amplitude in equal-temperament pitch bands
	text(1,ax[3]*.9, "ET bands nontuned RMS  = %f"%(F0.X[eq_bins]**2).mean()**0.5, fontsize=14)
	text(1,ax[3]*.8, "ET bands autotuned RMS = %f"%(F1.X[eq_bins]**2).mean()**0.5, fontsize=14)
	return mix0, mix1

def rms_spectrum_test(song='tainted', tuning_f0=110., channel=0):
	"""
	Extract spectral RMS power for equal temperament pitches
	inputs:
		song - directory name of song (contains: song/mix_000.wav and song/mix_100.wav non-autotuned and autotuned mixes)
		tuning_f0 - lowest frequency to track melody (110Hz = A440Hz/4) [110]
		channel - whether to use 0=left, 1=right, or 2=both channels [0] 
	outputs:
		dict {'nontuned_rms':df0, 'autotuned_rms':df1} energy (RMS power) at ideal pitch tuning freqs
	"""
	x0, sr, fmt = wavread(song+os.sep+'mix_000.wav')
	x1, sr, fmt = wavread(song+os.sep+'mix_100.wav')
	if channel==2: # mix the channels
		if len(x0.shape) > 1:
			x0 = x0.mean(1)
		if len(x1.shape) > 1:
			x1 = x1.mean(1)
	else: # extract given channel
		if len(x0.shape) > 1:
			x0 = x0[:,channel]
		if len(x1.shape) > 1:
			x1 = x1[:,channel]
	# Short-time Fourier analysis
	F0 = LinearFrequencySpectrum(x0,nfft=8192,wfft=8192,nhop=2048)
	F1 = LinearFrequencySpectrum(x1,nfft=8192,wfft=8192,nhop=2048)
	eq_freqs = tuning_f0*2**(arange(0,5,1/12.))
	eq_bins = array([argmin(abs(F0._fftfrqs-f)) for f in eq_freqs])
	# df0 = normalize(F0.X)[eq_bins].mean(1)
	df0 = (normalize(F0.X)[eq_bins]**2).mean(1)**0.5	
	#df1 = nomalize(F1.X)[eq_bins].mean(1)
	df1 = (normalize(F1.X)[eq_bins]**2).mean(1)**0.5
	figure()
	semilogx(F0._fftfrqs[eq_bins], df0)
	semilogx(F0._fftfrqs[eq_bins], df1)
	legend(['Original vocals','Autotuned vocals'],loc=0)
	title(song+': ET bands untuned/tuned vocals mixed with background', fontsize=20)
	xlabel('Equal Temperament Bands (Hz)',fontsize=20)
	ylabel('Power',fontsize=20)	
	grid()
	return {'nontuned_rms':rms_flat(df0), 'autotuned_rms':rms_flat(df1)}

def predominant_melody_test(song='tainted', tuning_f0=110., channel=0):
	"""
	Extract predominant melody (f0 track) and compare to equal temperament tuning.
	inputs:
		song - directory name of song (contains: song/mix_000.wav and song/mix_100.wav non-autotuned and autotuned mixes)
		tuning_f0 - lowest frequency to track melody (110Hz = A440Hz/4) [110]
		channel - whether to use 0=left, 1=right, or 2=both channels [0] 
	outputs:
		dict {'nontuned_deltas':df0, 'autotuned_deltas':df1} deviations from ideal pitch tuning
	"""
	p0 = PredominantMelody(frameSize=4096, hopSize=2048, 
        minFrequency=80.0, maxFrequency=20000., guessUnvoiced=True, voiceVibrato=False)
	p1 = PredominantMelody(frameSize=4096, hopSize=2048, 
        minFrequency=80.0, maxFrequency=20000., guessUnvoiced=True, voiceVibrato=False)
	x0, sr, fmt = wavread(song+os.sep+'mix_000.wav')
	x1, sr, fmt = wavread(song+os.sep+'mix_100.wav')
	if channel==2: # mix the channels
		if len(x0.shape) > 1:
			x0 = x0.mean(1)
		if len(x1.shape) > 1:
			x1 = x1.mean(1)
	else: # extract given channel
		if len(x0.shape) > 1:
			x0 = x0[:,channel]
		if len(x1.shape) > 1:
			x1 = x1[:,channel]
	mel00 = p0(array(x0,dtype='f'))[0]
	mel10 = p1(array(x1,dtype='f'))[0]
	eq_freqs = tuning_f0*2**(arange(0,5,1/12.))
	df0 = median([min(abs(eq_freqs-f)) for f in mel00[where(mel00)]])
	df1 = median([min(abs(eq_freqs-f)) for f in mel10[where(mel10)]])
	return {'nontuned_deltas':df0, 'autotuned_deltas':df1}

