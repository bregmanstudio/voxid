"""
voweltimbre.py - module for comparative analysis of sung timbres in a musical audio context.
	Evaluate extraction of sung vowel formants compared to a ground truth mark-up

	usage:
	(r,p), Z, Z2 = test_vowel_analysis(harmonic=True,tsne=True,dfw=True,perplexity=3)
	inputs:
		harmonic - whether to use predominant harmonic analysis (sung melody extraction)
    	tsne     - whether to return t-SNE or just DTW spectrum [True]  
    	dfw      - whether to perform dynamic FREQUENCY warping (dfw) after DTW [True]    
    	perplexity - how many near neighbours in t-SNE [3]
	outputs:
		(r,p) = pearson correlation coefficient and p-value 
	    Z = ground truth (markup) vowel frequencies embedding
	   Z2 = extracted vowel features embedding

Author: Michael A. Casey - Bregman Media Labs, Dartmouth College, Hanover, USA
Copyright (C) 2015, Dartmouth College, All Rights Reserved
License: MIT 
"""
from bregman import distance, testsignal, features
from pylab import *
import glob, csv, pdb
import scipy.stats as ss
try:
    from essentia.standard import *
except:
    print "warning: essentia module not found"
try:
    import dpcore
except:
    print "warning: no dynamic programming module"
try:
    from tsne import bh_sne
except:
    print "warning: no tsne module"

def predominant_harmonics(audio=[], filename='DKW_EttaJames_1961.wav', minFreq=200., maxFreq=2000., 
	SR=44100., N=4096, H=2048):
    """
    Extract the predominant harmonic spectrum from a polyphonic mixture
    inputs:
	    audio - the audio data [empty=use filename]
        filename - file to analyze ['DKW_EttaJames_1961.wav']
        minFreq  - minimum frequency for harmonic peaks [200]
        maxFreq  - maximum frequency for harmonic peaks [2000]
        SR, N, H - fft spectrum parameters
    outputs:
        pool - an essentia pool structure containing:
	        freqs - predominant harmonics frequencies
	         mags - predominant harmonics magnitudes
    """
    SR = float(SR)
    if not len(audio):
    	loader = essentia.standard.MonoLoader(filename=filename)
    	audio = loader()
    pool = essentia.Pool()
    predominant_melody = PredominantMelody(frameSize=N, hopSize=H, 
    	minFrequency=80.0, maxFrequency=20000., guessUnvoiced=False, voiceVibrato=True)
    p_melody = predominant_melody(audio)
    w = Windowing(type = 'hann')
    pkRange, minPos, maxPos = _calcPeakRange(SR, N, minFreq, 10000.)
    spectrum = Spectrum()
    peaks = PeakDetection(minPosition=minPos, maxPosition=maxPos, 
    	range=pkRange, maxPeaks=100, threshold=-100.)
    #harmonic_peaks = HarmonicPeaks(maxHarmonics=20) # this essentia class is broken, segfaults due to array error
    for i, frame in enumerate(FrameGenerator(audio, frameSize = N, hopSize = H)):
        spec = spectrum(w(frame))
        pk = peaks(20*log10(spec+finfo(float).eps))
        hpk = get_harmonic_peaks(pk[0] * (SR / N), 10**(pk[1]/20.), p_melody[0][i]) # python harmonic peaks, below
        pool.add('freqs', hpk[0])
        pool.add('mags', hpk[1])        
    pool.add('melody', p_melody[0])
    pool.add('melody_confidence', p_melody[1])
    pool.add('fname', filename.split('.')[0].replace('DKW_',''))
    return pool

def get_harmonic_peaks(freqs, mags, f0, maxHarmonics=20, tolerance=.2):
	"""
	Select harmonic peaks of given f0 from given spectral peak freqs and mags
	This is a Python port of the essentia harmonic peaks algorithm
	inputs:
		freqs - list of peak frequencies
		mags  - list of peak magnitudes
		f0    - estimated predominant fundamental frequency
		maxHarmonics - how many harmonics to return [20]
		tolerance - proportion of frequency deviation to allow (0,0.5) [0.2]
	outputs:
		harmonic_freqs, harmonic_mags - vectors of harmonic peak frequencies and magnitudes
	"""
	f0 = float(f0)	# just to make sure
	if tolerance<0.0 or tolerance>0.5:
		raise ValueError("HarmonicPeaks: tolerance must be in range (0,0.5)")
	if maxHarmonics<0 or maxHarmonics>100:
		raise ValueError("HarmonicPeaks: maxHarmonics must be in range [0,100]")
	if f0<0:
		raise ValueError("HarmonicPeaks: input pitch must be greater than zero")
	if len(freqs) != len(mags):
		raise ValueError("HarmonicPeaks: frequency and magnitude input vectors must have the same size")
	if f0 == 0: # pitch is unknown -> no harmonic peaks found
		return array([], dtype='f4'),array([], dtype='f4')
	if len(freqs)==0: # no peaks -> no harmonic peaks either
		return array([], dtype='f4'),array([], dtype='f4')
	if freqs[0] <= 0:
		raise ValueError("HarmonicPeaks: spectral peak frequencies must be greater than 0Hz")
	for i in xrange(1, len(freqs)):
		if freqs[i] < freqs[i-1]:
			raise ValueError("HarmonicPeaks: spectral peaks input must be ordered by frequency")
		if freqs[i] == freqs[i-1]:
			raise ValueError("HarmonicPeaks: duplicate spectral peak found, peaks cannot be duplicated")
    	if freqs[i] <= 0:
    		raise ValueError("HarmonicPeaks: spectral peak frequencies must be greater than 0Hz")
	ratioTolerance = tolerance
	candidates = [(-1,0)] * maxHarmonics # immutable for safety (assign new for replacement)
	ratioMax = maxHarmonics + ratioTolerance
	for i in xrange(len(freqs)):
		ratio = freqs[i] / f0
		harmonicNumber = int(round(ratio))
		distance = abs(ratio - harmonicNumber)
		if harmonicNumber < maxHarmonics: # Added by MKC 5/2/15, this is the ESSENTIA BUG
			if distance <= ratioTolerance and ratio <= ratioMax:
				if candidates[harmonicNumber-1][0] == -1 or distance < candidates[harmonicNumber-1][1]:
	    			# first occured candidate or a better candidate for harmonic
					candidates[harmonicNumber-1] = (i,distance)
			elif distance == candidates[harmonicNumber-1][1]:
	    		# select the one with max amplitude
				if mags[i] > mags[candidates[harmonicNumber-1][0]]:
					candidates[harmonicNumber-1] = (i, distance)
	harmonicFrequencies = []
	harmonicMagnitudes = []		
	for h in range(maxHarmonics):
		i = candidates[h][0]
		if i < 0:
			# harmonic not found, output ideal harmonic with 0 magnitude
			harmonicFrequencies.append((h+1) * f0)
			harmonicMagnitudes.append(0.)
		else:
			harmonicFrequencies.append(freqs[i])
			harmonicMagnitudes.append(mags[i])
	return array(harmonicFrequencies, dtype='f4'), array(harmonicMagnitudes, dtype='f4')

def _msg(s, newln=True):
	"""
	print messages instantly (flush stdout buffer)
	"""
	print s,
	if newln:
		print
	sys.stdout.flush()

def peaks_cqft(fname, freqs, mags, sonify=False, conv=False, N=4096, H=2048, SR=44100.):
	"""
	Peaks to constant-Q transform, or audio
	inputs:
		fname - name of audio file
		freqs - peak freqs per time-point
		mags  - peak magnitudes
		sonify- return audio instead of cqft [False]
		N     - fft length
		H     - fft hop
		SR    - audio sample rate
	outputs:
		bregman.LogFrequencySpectrum
	"""
	F = features.LogFrequencySpectrum(fname, nfft=N, wfft=N, nhop=H)
	Xhat = zeros(F.X.shape)	
	T = Xhat.shape[1]
	for t in xrange(len(freqs)):
		if t<T:
			for f,a in zip(freqs[t],mags[t]):
				Xhat[argmin(abs(F._logfrqs-f)),t]=a
	if conv:
		for t in xrange(len(freqs)):
			if t<T:
				Xhat[:,t] = convolve(Xhat[:,t], testsignal.gauss_pdf(10,5,1),'same')
	if sonify:
		Xhat = F.inverse(Xhat) # use original phases
	return Xhat.T

def peaks_sinusoids(freqs, mags, N=4096, H=2048, SR=44100.):
	"""
	Sinusoidal resynthesis of spectral peaks
	inputs:
		freqs - peak freqs per time-point
		mags  - peak magnitudes
		N     - fft length
		H     - fft hop
		SR    - audio sample rate
	outputs:
		audio
	"""
	SR = float(SR)
	phases = rand(100)*2*pi-pi # initial phase
	z = zeros(len(freqs)*H+N-H, dtype='f')
	for t, (freqs,mags) in enumerate(zip(freqs,mags)):
		x = zeros(N)
		for k in xrange(len(freqs)): 
			x += mags[k]*testsignal.sinusoid(f0=freqs[k], sr=SR, num_points=N, phase_offset=phases[k])
			phases[k] = (phases[k] + pi + N/SR * 2 * pi * freqs[k]) % (2*pi) - pi
		z[t*H:t*H+N] += hamming(N)*x
	return z

def _calcPeakRange(sr,n, loFreq=200., hiFreq=2000.):
	"""
	_calcPeakRange - utility function to calculate parameters for peaks	
	inputs:
		sr - sample rate
		n  - fft length
	  loFreq - lowest peak frequency
	  hiFreq - highest peak frequency
	"""
	fftBase = sr / float(n)
	pkRange = n/2 + 1
	minPos = int(round(loFreq/fftBase))
	maxPos = int(round(hiFreq/fftBase))
	return pkRange, minPos, maxPos

def _dct_coefs(N):    
    """
    Create discrete cosine transform coefficients for N x N matrix
    """
    d = array([cos(pi/N*(arange(N)+0.5)*k) for k in arange(N)],dtype='f4')
    d[0] *= 1/sqrt(2)
    d *= sqrt(2.0/N)
    return d

def _imagesc(X):
    """
    Emulate Matlab's imagesc() function
    """
    imshow(X, aspect = 'auto', origin='bottom', interpolation='nearest')
    colorbar()

def vowel_analysis(audio=[], filename='DKW_EttaJames_1961.wav', nBands=48, nCoefs=10, N=4096, H=2048):
    """
    Extract vowel analysis as frame#, F1, F2, F3 
    inputs:
        audio - the audio data [empty=use filename]
        filename - audio filename ['DKW_EttaJames_1961.wav']
        nBands - spectrum bands per octave [48]
        nCoefs - cepstrum number of coefficients to retain [10]
    outputs:
        pool - an essentia pool structure containing:
           mel_bands - Mel frequency bands
           mel_coefs - Mel cepstral coefficients (direct DCT)
           mfccs     - Mel cepstral coefficients (essentia)
           lqft      - inverse of direct DCT mel_coefs
           peaks     - peaks, sorted by frequency, of lqft 
    """
    if not len(audio):
    	loader = essentia.standard.MonoLoader(filename=filename)
    	audio = loader()
    pool = essentia.Pool()
    w = Windowing(type = 'hann')
    spectrum = Spectrum()
    mfcc = MFCC(inputSize=N/2+1)
    melbands = MelBands(inputSize=N/2+1, numberBands=nBands)
    D = _dct_coefs(nBands)[:nCoefs].T
    peaks = PeakDetection(range=nBands, minPosition=0, maxPosition=nBands-1, maxPeaks=10)
    for frame in FrameGenerator(audio, frameSize = N, hopSize = H):
        spec = spectrum(w(frame))
        mel_bands = 20*log10(melbands(spec+finfo(float).eps))
        mel_coefs = dot(D.T,mel_bands)
        pool.add('mel_bands', mel_bands) 
        pool.add('mel_coefs', mel_coefs)
        pool.add('mfccs', mfcc(spec)[1])
        lqft = dot(D, mel_coefs)
        pool.add('peaks',peaks(lqft)[0])
#       lqft = 10**(lqft/20.)
#       lqft = lqft / lqft.max()
        pool.add('lqft', lqft)
    if filename is not None:        
	    pool.add('fname', filename.split('.')[0].replace('DKW_',''))
    return pool

def _plot_vowels(pool, saveplot=False):
    mel_bands = essentia.array(pool['mel_bands']).T
    mel_coefs = essentia.array(pool['mel_coefs']).T
    mfccs = essentia.array(pool['mfccs']).T
    lqft = essentia.array(pool['lqft']).T
    peaks = pool['peaks']

    if(plotting):
        figure()
        # Mel Bands
        subplot(221)
        _imagesc(mel_bands)
        title('Mel Freq. Bands',fontsize=14)
        # LQFT
        subplot(222)
        _imagesc(lqft)
        title('Low Quefrency, Peaks', fontsize=14)
        # Show Peaks
        [plot(t,qq,'bo') for t,q in enumerate(peaks) for qq in q]
        axis('tight')
        # MFCCs
        subplot(223)
        _imagesc(mfccs[1:,:])
        title('13 MFCC Coefs', fontsize=14)
        # DCT Projected MFCCs
        subplot(224)
        _imagesc(mel_coefs[1:,:])
        nCoefs = mel_coefs.shape[0]
        title('%d MFCC Coefs'%nCoefs, fontsize=14)
        suptitle(pool['fname'][0],fontsize=16)
        if saveplot:
            savefig('%s.png'%pool['fname'][0])

def _save_data(pool, key='lqft', delim=','):
        if savedata:
            with open('%s.%s'%(pool['fname'][0],key), 'wt') as f:
                for p in pool[key]:
                    for i, a in enumerate(p):
                        if i != 0:
                            fwrite(delim)
                        f.write(str(a))
                    f.write('\n')

def align_vowels(X,Y, dfun=distance.euc_normed):
    """
    DTW align vowels based on spectral data in X and Y
    Uses dpcore dynamic programming library
    inpus:
      X, Y - spectral data in rows (num columns must be equal)
    outputs:
      p, q, c - Dynamic Time Warp coefs: X[p]<-->Y[q], c=cost
    """
    Df = dfun(X,Y)
    p,q,C,phi = dpcore.dp(Df, penalty=0.1, gutter=0.1)
    return {'p':p,'q':q,'C':C}

def analyze_all(V, dfun=distance.euc_normed, dfw=True, **kwargs):
    """
    Perform DTW(+DFW) timbre analysis on list of audio files, return pair-wise costs
    inputs:
       V       - vowel analysis pool
       dfun    - distance function to use [bregman.distance.euc_normed]
       dfw     - whether to perform dynamic FREQUENCY warping (dfw) after DTW [True]
       **kwargs  - arguments to dpcore.dp DTW algorithm
    outputs:
       Z - pair-wise cost matrix (timbre time/frequency warp) between audio files
    """
    Z = zeros((len(V),len(V)))
    kwargs['penalty'] = kwargs.pop('penalty',0.1)    
    kwargs['gutter'] = kwargs.pop('gutter',0.0)
    for i,a in enumerate(V):
        for j,b in enumerate(V):
        	D = dfun(a['mfccs'],b['mfccs'])
        	p,q,c,phi = dpcore.dp(D, **kwargs)
        	if not dfw:
        		Z[i,j] = diag(dfun(a['lqft'][p],b['lqft'][q])).mean()
        	else:
	        	alpha = optimal_vtln(b['lqft'][q], a['lqft'][p], 'symmetric')
	        	print "Optimal DFW (VTLN) warp alpha=", alpha
	        	Vi_warped = vtln(b['lqft'][q], 'symmetric', alpha)
	        	Z[i,j] = diag(dfun(Vi_warped,b['lqft'][q])).mean()
    return Z

def tsne_all(V, ref_idx=9, dfun=distance.euc_normed, dfw=True, tsne=True, perplexity=3, plotting=False, **kwargs):
    """
    Perform tsne / DTW / DFW timbre analysis on list of audio files, return pair-wise costs
    Uses tsne library
    inputs:
       V 	   - vowel analysis pool
    ref_idx    - reference audio file, all audio warped to this [9]
       dfun    - distance function to use [bregman.distance.euc_normed]
       dfw     - whether to perform dynamic FREQUENCY warping (dfw) after DTW [True]    
       tsne    - whether to return t-SNE or just DTW spectrum [True]  
    perplexity - how many near neighbours in t-SNE [3]
    plotting   - whether to plot the t-SNE result [False]
       **kwargs  - arguments to dpcore.dp DTW algorithm    
    outputs:
       Z - 2D projection map of audio files
    """
    Z = []
    kwargs['penalty'] = kwargs.pop('penalty',0.1)    
    kwargs['gutter'] = kwargs.pop('gutter',0.0)
    for i in xrange(len(V)):
        C = dfun(V[ref_idx]['mfccs'],V[i]['mfccs'])
        p,q,c,phi = dpcore.dp(C, **kwargs)
        if not dfw:
            Z.append(V[i]['lqft'][q])
        else:
            alpha = optimal_vtln(V[i]['lqft'][q], V[ref_idx]['lqft'][p], 'symmetric')
            print "Optimal DFW (VTLN) warp alpha=", alpha
            Vi_warped = vtln(V[i]['lqft'][q], 'symmetric', alpha)
            Z.append(Vi_warped)
    if tsne:
        X = array([zz.flatten() for zz in Z], dtype='f8')
        Z = _tsne(X, perplexity=perplexity, plotting=plotting)
    return Z

def optimal_vtln(Y,X, warpFunction='asymmetric'):
    """
    Return optimal frequency warp between spectrum data Y and target spectrum X.
    inputs:
       Y - spectrogram data to warp
       X - target spectrogram
       warpFunction - which warp method to use ['asymmetric']
    outputs:
       alpha - optimal warp factor
    """
    min_mse = inf
    for alpha in arange(0.1,1.8,.1):
        Xhat = vtln(Y, warpFunction, alpha)
        mse = ((X - Xhat)**2).mean()
        if mse < min_mse:
            min_mse = mse
            min_alpha = alpha
    print "alpha=%.3f, min_mse=%.3f"%(min_alpha,min_mse)
    return min_alpha

def vtln(frames, warpFunction='asymmetric', alpha=1.0):
    """
    Vocal tract length normalization via frequency warping
    Python port of David Sundermann's matlab implementation by M. Casey
    inputs:
       frames - the frequency data to warp
       warpFuction - asymmetric, symmetric, power, quadratic, bilinear [asymmetric]
       alpha - the warp factor
    """
    warp_funs = ['asymmetric', 'symmetric', 'power', 'quadratic', 'bilinear']
    if not warpFunction in warp_funs:
        print "Invalid warp function"
        return
    warpedFreqs = zeros(frames.shape)
    for j in xrange(len(frames)):
        m = len(frames[j])
        omega = (arange(m)+1.0) / m * pi
        omega_warped = omega
        if warpFunction is 'asymmetric' or warpFunction is 'symmetric':
            omega0 = 7.0/8.0 * pi
            if warpFunction is 'symmetric' and alpha > 1:
                omega0 = 7.0/(8.0 * alpha) * pi
            omega_warped[where(omega <= omega0)] = alpha * omega[where(omega <= omega0)]
            omega_warped[where(omega > omega0)] = alpha * omega0 + ((pi - alpha * omega0)/(pi - omega0)) * (omega[where(omega > omega0)] - omega0)
            omega_warped[where(omega_warped >= pi)] = pi - 0.00001 + 0.00001 * (omega_warped[where(omega_warped >= pi)])
        elif warpFunction is 'power':
            omega_warped = pi * (omega / pi) ** alpha
        elif warpFunction is 'quadratic':
            omega_warped = omega + alpha * (omega / pi - (omega / pi)**2)
        elif warpFunction is 'bilinear':
            z = exp(omega * 1j)
            omega_warped = abs(-1j * log((z - alpha)/(1 - alpha*z)))
        omega_warped = omega_warped / pi * m
        warpedFrame = interp(omega_warped, arange(m)+1, frames[j]).T
        if isreal(frames[j][-1]):
            warpedFrame[-1] = real(warpedFrame[-1])
        warpedFrame[isnan(warpedFrame)] = 0
        warpedFreqs[j]=warpedFrame
    return warpedFreqs

def tsne_ground_truth(file="StormyWeather_DataSet.csv", dtype='f8', tsne=True, **kwargs):
	"""
	Load vowel formant frequency ground truth labels and create array of freqs
	"""
	vowels = []
	with open(file,"r") as f:
		reader = csv.reader(f)
		C = [row for row in reader]
		vowels.append(array([c[2:5] for c in C[3::6]],dtype=dtype))
		vowels.append(array([c[2:5] for c in C[4::6]],dtype=dtype))
		vowels.append(array([c[2:5] for c in C[5::6]],dtype=dtype))
		vowels.append(array([c[2:5] for c in C[6::6]],dtype=dtype))
	vowels = array(zip(*vowels)).reshape(15,-1)
	if not tsne:
		return vowels
	else:
		Z = _tsne(vowels, **kwargs)
		return Z

def _tsne(X, dir_str="*.wav", perplexity=3, plotting=False):
	"""
	Utility function to compute tsne
	"""
	flist = sorted(glob.glob(dir_str))
	Z = bh_sne(X, perplexity=perplexity)
	if plotting:
		figure()
		plot(Z[:,0], Z[:,1],'r.')
		[[text(p[0],p[1],'%s'%flist[i],fontsize=12) for i,p in enumerate(Z)]]
	return Z

def harmonic_tsne_all(dir_expr="*.wav", return_analyses=False, **kwargs):
	"""
	time and frequency warped t-SNE of predominant harmonic spectrum
	inputs:
		dir_expr - directory expression for audio files ["*.wav"]
	  return_analyses - return intermediate analyses instead of t-SNE or feature vector space
		**kwargs - key-word arguments for tsne analysis function {harmonic_}tsne_all()
			ref_idx    - reference audio file, all audio warped to this [9]
       		dfun    - distance function to use [bregman.distance.euc_normed]
    		dfw     - whether to perform dynamic FREQUENCY warping (dfw) after DTW [True]    
    		tsne    - whether to return t-SNE or just DTW spectrum [True]  
    		perplexity - how many near neighbours in t-SNE [3]
    		plotting   - whether to plot the t-SNE result [True]
    		**kwargs  - arguments to dpcore.dp DTW algorithm   
	outputs (depends on value of return_analyses):
		Z2 - 2d embedding of dtw-dfw aligned predominant harmonics vowel analysis [return_analyses=False]
		H, X, V - harmonic peaks, reconstructed signals, harmonic vowel analysis spectra [return_analyses=True]
	"""
	normalize = testsignal.balance_signal
	flist = sorted(glob.glob(dir_expr))
	conv = kwargs.pop("conv",False)
	H = [predominant_harmonics(filename=f, N=4096, H=1024) for f in flist]
	X = [normalize(peaks_cqft(f, H[i]['freqs'], H[i]['mags'], conv=conv, sonify=True, N=4096,H=1024)) for i,f in enumerate(flist)]
	V = [vowel_analysis(array(x,dtype='f4'),f) for x,f in zip(X,flist)]
	Z2 = tsne_all(V,ref_idx=9,**kwargs)	
	if return_analyses:
		return H, X, V 
	return Z2

def test_vowel_analysis(dir_expr="*.wav",harmonic=True,null_model=False,**kwargs):
	"""
	pearson r test bewteen vowel analysis and ground truth markup
	inputs:
		harmonic - whether to use predominant harmonic spectrum [True]
		null_model - whether to permute ground truth indices for null model testing
		**kwargs - key-word arguments for tsne analysis function {harmonic_}tsne_all()
			ref_idx    - reference audio file, all audio warped to this [9]
       		dfun    - distance function to use [bregman.distance.euc_normed]
    		dfw     - whether to perform dynamic FREQUENCY warping (dfw) after DTW [True]    
    		tsne    - whether to return t-SNE or just DTW spectrum [True]  
    		perplexity - how many near neighbours in t-SNE [3]
    		plotting   - whether to plot the t-SNE result [True]
    		**kwargs  - arguments to dpcore.dp DTW algorithm   
    outputs:
    	(r,p) - pearson r coefficient and p-value
    	    Z - ground truth embedding [tsne=True] / features [tsne=False]
    	   Z2 - vowel analysis embedding [tsne=True] / features [tsne=False]
	"""
	Z = tsne_ground_truth(**kwargs)
	if null_model:
		Z = permutation(Z)
	if harmonic:
		Z2 = harmonic_tsne_all(dir_expr=dir_expr, **kwargs)
	else:
		flist = sorted(glob.glob(dir_expr))
		V = [vowel_analysis(filename=f, N=4096, H=1024) for f in flist]
		Z2 = tsne_all(V, ref_idx=9,**kwargs)
	Z2 = array([z.flatten() for z in Z2])
	D0, D1 = distance.euc(Z,Z), distance.euc(Z2,Z2)
	# Remove zeros on main diagonal
	d0, d1 = D0[where(1-eye(len(Z)))].flatten(), D1[where(1-eye(len(Z2)))].flatten()
	p = ss.pearsonr(d0,d1)
	return p, Z, Z2

def ismir2015_evaluate(niter=0, **kwargs):
	"""
	inputs:
		niter - number of permutation test iterations to run [0=none]
		**kwargs - key-word arguments for tsne analysis function {harmonic_}tsne_all()
			ref_idx    - reference audio file, all audio warped to this [9]
       		dfun    - distance function to use [bregman.distance.euc_normed]
    		dfw     - whether to perform dynamic FREQUENCY warping (dfw) after DTW [True]    
    		tsne    - whether to return t-SNE or just DTW spectrum [True]  
    		perplexity - how many near neighbours in t-SNE [3]
    		plotting   - whether to plot the t-SNE result [True]
    		**kwargs  - arguments to dpcore.dp DTW algorithm   
	outputs:
		res_n, res_h - result dicts for non-harmonic or harmonic vowel analyses {'p','Z','Z2'}
	"""
	res_n = {} # non harmonic analysis	(full spectrum)
	res_n['p'],res_n['Z'],res_n['Z2'] = test_vowel_analysis(harmonic=False,**kwargs)
	res_n['null'] = [test_vowel_analysis(harmonic=False,null_model=True,**kwargs) for _ in range(niter)]
	res_h = {} # predominant harmonics analysis (extracted voice)
	res_h['p'],res_h['Z'],res_h['Z2'] = test_vowel_analysis(harmonic=True,**kwargs)
	res_h['null'] = [test_vowel_analysis(harmonic=True,null_model=True,**kwargs) for _ in range(niter)]
	return res_n, res_h

if __name__ == "__main__":
	# ISMIR 2015 test
	"""
       Dynamic frequency warping VTLN
       Test with ground truth markup (perasonr plus permution null model)
       Background sound reduction [PreFest]
       Results per vowel (via alignment to reference mark-up)
       Plot tsne per vowel

       Currently, tsne on predominant harmonics is seemingly random. Flip-flopping between correlated and not with ground truth.
       Suggest inspecting vowel spectrum representation. Then experimenting with dimensionality reduction in tsne (pca_d), although this crashes at the moment. 
       
	"""
	p_n, Z_n, Z2_n = test_vowel_analysis(harmonic=False,tsne=True,dfw=True,perplexity=3)	
	p_h, Z_h, Z2_h = test_vowel_analysis(harmonic=True,tsne=True,dfw=True,perplexity=3)
	
