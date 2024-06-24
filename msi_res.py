#! python3 
""" Author - Martin Metodiev. Scripts for calculating spatial resolution in MSI. Used for thesis."""
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy import fftpack
from scipy import special
from scipy import optimize
from scipy import integrate
from scipy import signal
import lmfit

from skimage.segmentation import watershed
from skimage.filters import sobel

#import tvregdiff


np.random.seed(16)


#define gaussian cdf
esf = lambda x, sigma, mu, I: (0.5)*(I)*(1 + special.erf((x-mu)/(np.sqrt(2)*sigma)))

#define gaussian cdf with I0 and I1
esf_p = lambda x, sigma, mu, I1, I0: I0  + (0.5)*(I1 - I0)*(1 + special.erf((x-mu)/(np.sqrt(2)*sigma)))


#define gaussian cdf with two parameters only (sigma and mu)

def esf_simple(x, sigma, mu):
    return (0.5)*(1 + special.erf((x-mu)/(np.sqrt(2)*sigma)))

        
#define normalised gaussian
def gaussian(x, sigma, A):
    return A*np.exp(-((x)**2)/(2*sigma**2))

def gaussian_mu(x, sigma, A, mu):
    return A*np.exp(-((x-mu)**2)/(2*sigma**2))
 




#define a function that sums over ROI linescans in the y direction
def esf1d(data):
    
    """sum over a 2d esf and obtain a 1d average; don't subtract"""
    
    data_avg = np.mean(data, axis = 0)
    
    
    return data_avg




#define a function that takes a 1D esf and returns a 1D LSF
def lsf1d_noisy(data, alpha=23e-1):
    
    """ differentiate a 1d est to obtain a 1d lsf in the presence of noise. returns a normalized value. input must be 1d! based on  https://doi.org/10.5402/2011/164564"""
    
    #lsf_avg = np.gradient(data)
    
    lsf_avg = tvregdiff.TVRegDiff(data, itern=1, alph=alpha, scale='small', precondflag=False, plotflag=False, diffkernel='abs' )
    
    lsf_avg = lsf_avg/len(lsf_avg)

    return lsf_avg
    
def lsf1d(data):
    
    """ standard derivative. it is used for high intensity signals where noise is less dominant"""
    
    lsf_avg = np.gradient(data)
    

    return lsf_avg


#define a function that computes an MTF from a 1D LSF
def mtf(lsf1d):
    
    """compute 1d mtf - absolute value of 1d lsf
    Input: 1d lsf
    output: mtf (abs value of FT of 1d lsf)
    """
    
    mtf = np.abs(((fftpack.fftshift(fftpack.fft(fftpack.ifftshift(lsf1d))))))
    
    return mtf

#define a function that fits gaussian cdf to to ESF
def esf_fit(data):

    """use scipy's curve_fit to fit an ESF to data
    Input: 1d ESF
    
    Otput: two 4x1 arrays.
    First array contains the parameter estimates - sigma (blur), mu (centre of esf), I1 (surface intensity), I0.
    The second array contains the errors
    
    """
    
    xc = np.argmax(np.gradient(data)) #guess where xc is
    
    Ibar = np.average(data[xc:]) #guess what intensity is by computing average 
    
    #I0 = np.average(data[:xc])
    
    x = np.linspace(0,data.shape[0]-1, data.shape[0]) #create an array with data points (pixels) in x dir
    
    param, cov = curve_fit(esf_p, x, data, p0 = [0.5,0, Ibar, 0], maxfev = 5000) #fit funtion with curve_fit; bootstrap seems to do better
    
    #pstart = [0.5,0, Ibar, 0]
    #param_boot, err_boot = fit_bootstrap(pstart, x, data, ff_esf)

    
    err = np.sqrt(np.diag(cov)) # compute errors from covariance matrix
    
    return param, err #param_boot, err_boot - use for bootstrap fitting


def esf_fit_simplified(data):

    """use lmfit to fit an ESF to data. The data is normalised to a simplified gaussian CDF (no I1, I0). Reducing the number of parameters improves the error.
    Errors are not returned!
    
    Input: 1d ESF
    
    Otput: one 5x1 array. 
    1. is the sigma (blur) estimate.
    2. mu.
    3. median value of the surface intensity (I1)
    4. median value of the non-surface intensity (I0)
    5. reduced chi squared of fit
    
    """
    
    
    #data must be 1d
    xc = np.argmax(np.gradient(data)) #guess where xc is
    
    Ibar = np.average(data[xc:]) #guess what intensity is by computing average 
    
    #I0 = np.average(data[:xc])
    
    x = np.linspace(0,data.shape[0]-1, data.shape[0]) #create an array with data points (pixels) in x dir
    
    param, cov = curve_fit(esf_p, x, data, p0 = [0.5,0, Ibar, 0], maxfev = 5000) #fit funtion with curve_fit
    
    x_I1 = round(param[1] + 3*param[0]) + 3
    x_I0 = round(param[1] - 3*param[0]) - 3
    
    #print(x_I1, x_I0)
    
    #print(x_I1)
    #print(x_I0)
    data_simple = (data - np.median(data[:x_I0]))/(np.median(data[x_I1:])-np.median(data[:x_I0]))
    
    #print(x_I1)
    
    gmodel = lmfit.Model(esf_simple)
    params = gmodel.make_params(sigma=0.7, mu=20)
    
    
    result = gmodel.fit(data_simple, params, x=x)
    
    est_params = np.array(result.params)
    red_chi = result.redchi
    
    
    #err = np.sqrt(np.diag(cov)) # compute errors from covariance matrix
    
    return est_params[0], est_params[1], np.median(data[x_I1:]), np.median(data[:x_I0]), red_chi 

def ff(x, p):
    return gaussian(x, *p)

def ffmu(x, p):
    return gaussian_mu(x, *p)
    
def ff_esf(x, p):
    return esf_p(x, *p)


def fit_bootstrap(p0, datax, datay, function, yerr_systematic=0.0):

    """ bootstrap fitting. 
    code taken from https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i
    """

    errfunc = lambda p, x, y: function(x,p) - y

    # Fit first time
    pfit, perr = optimize.leastsq(errfunc, p0, args=(datax, datay), full_output=0)


    # Get the stdev of the residuals
    residuals = errfunc(pfit, datax, datay)
    sigma_res = np.std(residuals)

    sigma_err_total = np.sqrt(sigma_res**2 + yerr_systematic**2)

    # 100 random data sets are generated and fitted
    ps = []
    for i in range(100):

        randomDelta = np.random.normal(0., sigma_err_total, len(datay))
        randomdataY = datay + randomDelta

        randomfit, randomcov = \
            optimize.leastsq(errfunc, p0, args=(datax, randomdataY),\
                             full_output=0)

        ps.append(randomfit) 

    ps = np.array(ps)
    mean_pfit = np.mean(ps,0)

    # You can choose the confidence interval that you want for your
    # parameter estimates: 
    Nsigma = 1. # 1sigma gets approximately the same as methods above
                # 1sigma corresponds to 68.3% confidence interval
                # 2sigma corresponds to 95.44% confidence interval
    err_pfit = Nsigma * np.std(ps,0) 

    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit
    return pfit_bootstrap, perr_bootstrap 

	
def calculate_esf(data, pixel_size, disp=0, simplified=False):


    """ calculate the ESF parameters of an image.
    Input: step edge region of interest
    pixel_size - used when plotting
    disp=1 outputs a plot. default is zeros
    simplified = use the fit_esf_simplified function. Default is false. Use when interested in red chi squared statistics
    
    Returns:
    4x1 array
    1. sigma
    2. SNR
    3. surface intensity
    4. reduced chi squared of fit. If simplified=False this will be a nan
    
    """


    
    data_avg = esf1d(data) #turn into signal into 1d signal
    
    lsf = lsf1d(data_avg) # compute lsf
     
    xc = np.argmax(lsf) #get element corresponding to 50% of edge transition. need for surface calculations
    #print(xc) #use this to print x value of lsf max
    
    if simplified==False:
        p, err = esf_fit(data_avg)  #fit gaussian cdf to data
        # red_chi=np.nan
        ss_res = ((data - esf_p(np.arange(data_avg.size), *p))**2).sum()
        ss_tot = ((data - data_avg)**2).sum()
        r2 = 1 - ss_res/ss_tot
        #print(p[1]) #use this to print x value of 50% esf transition
        #print(np.sqrt(np.diag(err)))
    if simplified==True:
        p_sigma, p_mu, x_I1, x_I0, red_chi = esf_fit_simplified(data_avg)
        p = np.array([p_sigma, p_mu, x_I1, x_I0])
	
    #find where edge surface starts
    if xc > p[1]: # p[1] is the 50% x value from gaussian cdf fit ; xc is the x value of lsf maximum. theoretically they should match 
        xc = int(p[1] + 3) 
    else:
        xc = xc+1  # assume that surface starts  3 pixels away from x point corresponding to 50% of edge transition

    #print(xc) #use this to print x value of surface
    

    xc= xc+1 #1 8-for strong blur in thesis simulation!
    
    #print(xc)
    #x_I1 = round(p[1] + 3*p[0]) + 3
    
    ibar = np.mean(data_avg[xc:]) #compute average of surface
    #vbar = np.var(data_avg[xc:], ddof=1) #compute variance of surface
    
    #add graph plotter
    if disp != 0 :
        x = np.linspace(0, data_avg.shape[0]-1, data_avg.shape[0])
        plt.plot(x,data_avg, 'o-', label = 'data')
        plt.plot(x,lsf, 'o-', label = 'lsf')
        plt.plot(x,esf_p(x, p[0],p[1],p[2], p[3]), 'o-', label = 'gaussian cdf fit')
        plt.title('average signal at {} $\mu m$'.format(pixel_size))
        plt.xlabel('pixels')
        plt.ylabel('Ion count')
        plt.legend()
    
    #noise = np.zeros(data.shape)
    
    noise =  data - data_avg
        

    #noise = 1*np.sum(noise[:-1,:],axis=0)
    
    noisestd = np.std(noise[:,xc:])
    
    
    #print(noisestd)
    #noisestd = np.mean(noisestd)
    #calculate average snr
    snr = ibar/noisestd
    
    
    #print(p[0], snr) #print curve fit values plus snr
    #print(p[1])
    return p[0], snr, ibar, r2




#define a function that calculates an MTF from ESF ROI
def calculate_mtf(data, pixel_size, disp=0, med_filt=False, fit_to_fourier=False, noise_der=False, alpha=23e-1):

    """ calculate the MTF parameters of an image.
    Input: step edge region of interest
    
    pixel_size - used when plotting
    
    disp=1  - outputs a plot. default is zeros
    
    med_filt - apply a simple (3x3) median filter to data.
               Numerical differentiation can increase noise levels.
               This helps reduce this. Default is false.
               It will have no effect if fit_to_fourier=False.
    
    fit_to_fourier - fit the MTF to the FT of the LSF. Numerical differentiation of the edge can increase noise. Using this is not recommended for noisy images as this will sharpen the MTF and lead to a unnaturaly high resolution.
    
    noise_der - denoising methods can alter the width of the lsf. 
                this method applies a TV regularisation numerical differentiation approach based on https://www.hindawi.com/journals/isrn/2011/164564/.
                Python implementation taken from https://github.com/stur86/tvregdiff.
                Default is false. It will have no effect if fit_to_fourier=False.
    
    alpha - the alpha parameter for the TVRegdiff method.
    

    Returns:
    1. resolution frequency
    2. MTF cut off 
    3. resolution frequency error
    

    
    """
       
    data_avg = esf1d(data) #turn into signal into 1d signal
    


    #p_sigma, p_mu, x_I1, x_I0, red_chi = esf_fit_simplified(data_avg)  #fit gaussian cdf to data
    #p = np.array([p_sigma, p_mu, x_I1, x_I0])
    
    p, err = esf_fit(data_avg)
    
    #if 100*p[0]/err[0] > 15:
    #    print('Warning! Fractional error in esf blur parameter exceeds 15%! This will affect the resolution value!')
    

    ibar = np.mean(data_avg[int(p[1]+3):]) #calculate average intensity of surface

    
    x = np.linspace(0,data_avg.shape[0]-1, data_avg.shape[0]) #create array with pixels 
    
    esf_theoretical = esf_p(x,p[0], p[1], p[2], p[3]) #create a gaussian cdf using parameters extracted from data
    
    if noise_der==True and med_filt==False:
        lsf = lsf1d_noisy(data_avg, alpha)
    if noise_der==False and med_filt==False:
        lsf = lsf1d(data_avg)
    if noise_der==False and med_filt==True:
        sig_medfilt = esf1d(signal.medfilt2d(data, (3,3)))
        sig_medfilt[-1] = sig_medfilt[-2]
        lsf = lsf1d(sig_medfilt)
    if noise_der==True and med_filt==True:
        sig_medfilt = esf1d(signal.medfilt2d(data, (3,3)))
        sig_medfilt[-1] = sig_medfilt[-2]
        lsf = lsf1d_noisy(sig_medfilt, alpha)
        
        
    mtf_data = mtf(lsf)
    mtf_theoretical = mtf(np.gradient(esf_theoretical))   


    
    
    f = 2*fftpack.fftfreq(np.shape(mtf_data[0:])[0], d=1) #create frequency array
    
    pstart = [0.25, np.abs(integrate.simps(f, mtf_theoretical))]
    
    if fit_to_fourier==True:
        pfit, perr = fit_bootstrap(pstart, fftpack.fftshift(f), mtf_data, ff)
    else:
        pfit, perr = fit_bootstrap(pstart, fftpack.fftshift(f), mtf_theoretical, ff)
    
    

        
    #print(pfit, perr)    
    mtf_gauss = gaussian(fftpack.fftshift(f), pfit[0], pfit[1])

    mtfeq = interp1d(fftpack.fftshift(f), mtf_gauss, fill_value="extrapolate") #create an equation of mtf from gaussian fit by interpolating
    
    #noise = np.zeros(data.shape)
    #noise2 = np.zeros(data.shape)
    #noise_ft = np.zeros(data.shape)
    #noise_ft2 = np.zeros(data.shape)
    
    #for i in range(data.shape[0]):
    #    noise[i,:] = (data[i,:] -data_avg)
    #for i in range(data.shape[0]):
    #    noise_ft[i,:] = np.sqrt(np.abs(fftpack.fftshift(fftpack.fft(fftpack.ifftshift(noise[i,:]))))**2)
    
 

    #for i in range(data.shape[1]):
        #noise2[:,i] = (data[:,i] -np.mean(data, axis=0))
    #for i in range(data.shape[1]):
        #noise_ft2[:,i] = np.sqrt(np.abs(fftpack.fftshift(fftpack.fft(fftpack.ifftshift(noise2[:,i]))))**2)
    
    
    
    
    #nps = (1/(noise_ft.shape[0]-1))*np.sum(noise_ft,axis=0)/np.sqrt(data[:,int(p[1]+0.5):].shape[1]) #np.sqrt(data[:,int(p[1]+0.5):].shape[1])
        
    #fnoise = 2*fftpack.fftfreq(nps.shape[0], d=1)
    
    #noise = (np.sqrt(data.shape[0]))*(data_avg - esf_theoretical)
    
    xc = int(p[1]+0.5)
    #xc = xc + 10
    
    fnoise, PSD = signal.periodogram(data[:,xc:], 1, scaling='density')
    
    fnoise = 2*fnoise
    
    nps = np.sqrt(np.mean(PSD/2, axis=0))

    const = np.mean(nps[1:])
    
    noiseq = interp1d((fnoise), const*np.ones(fnoise.shape), fill_value="extrapolate")
    #npsinterp = interp1d(fftpack.fftshift(fnoise), nps, fill_value = "extrapolate")

    #noisestd = np.std(npsinterp(fnoise)[int(np.where(fftpack.fftshift(f) == 0)[0]):])
    #print(noisestd)
    noisestd = np.std(nps[1:])

    
   #fnoise_new = 2*fftpack.fftfreq(10*np.shape(noise[0:])[0], d=1)
    f_new = 2*fftpack.fftfreq(1000*np.shape(mtf_data[0:])[0], d=1) #create a new array of frequencies with that has a 100times more points

    
    x2 = np.isclose(noiseq(f_new), fftpack.fftshift(mtfeq(f_new)), atol=5e-3*ibar).astype(int) #find intersection point between MTF and NPS in y dir
    x2_ind = np.argmax(x2) #get array value of intersection point (x dir)
    f_cutoff_gauss = np.abs(fftpack.fftshift(f_new)[x2_ind]) #get frequency value corresponding to array value - this is cut off freq; resolution point
    spectral_cut_off = 100*const/mtf_gauss[np.where(fftpack.fftshift(f)==0)][0] #compute percantge of average NPS 
    
    if f_cutoff_gauss == 1.0:
        x2 = np.isclose(noiseq(f_new), fftpack.fftshift(mtfeq(f_new)), atol=5e-5*ibar).astype(int) #find intersection point between MTF and NPS in y dir
        x2_ind = np.argmax(x2) #get array value of intersection point (x dir)
        f_cutoff_gauss = np.abs(fftpack.fftshift(f_new)[x2_ind])
    
    ferr = np.sqrt( ((noisestd**2) *(pfit[0]**2)) / (2*noisestd**2 * np.log(pfit[1]/const)) + (2*perr[0]**2 * np.log(pfit[1]/const)) + ((perr[1]**2 * pfit[0]**2)/(2*pfit[1]**2 * np.log(pfit[1]/const))) )
    #print("test")
    #add plot
    if disp != 0 :

        plt.plot(fftpack.fftshift(f), mtf_data, label = 'data')
        plt.plot(fftpack.fftshift(f), mtf_gauss, label = 'gaussian')
        plt.plot((fnoise), nps, 'r')
        plt.plot(const*np.ones(fnoise.shape), 'r--', label = '$\sqrt{\dfrac{NSD}{2}}$' + '$={0:.1f} \%$'.format(spectral_cut_off))
        plt.xlim(0,1)
        plt.xlabel('Normalised frequency')
        plt.ylabel('MTF')
        plt.title('MTF at {} $\mu m$'.format(pixel_size))
        plt.legend()
        
    
    return f_cutoff_gauss, spectral_cut_off, ferr
    

def calc_res_dc(data_cube, roi1=np.index_exp[:,:], roi0=np.index_exp[:,:], sigma_err_tr=0.23, red_chi_tr=1.1,
    mtf_err_tr=0.15, sigma_tr=3, med_filt=False,fit_to_fourier=False, noise_der=False, alpha=23e-1):
    

    
    real_space = np.zeros((data_cube.shape[0], 4))
    fourier_space = np.zeros((data_cube.shape[0], 3))
    
    for i in range(data_cube.shape[0]):
        
        #ROI = np.rot90(data_cube[i][260:288,500:520],3)
        #surface = ROI[:,7:]
        #nosurf = ROI[:,:2]

        ROI = data_cube[i]

        surface = ROI[roi1] #[:,50:]
        nosurf = ROI[roi0] #[:,:30]
        

            
        ion_count_surf = np.mean(surface) #change to surface[i] when roi outside for
        ion_count_nosurf = np.mean(nosurf)
        
        if ion_count_surf> ion_count_nosurf:
            try:
                
                pars, errs = esf_fit(np.mean(ROI,axis=0))

                p_sigma, p_mu, x_I1, x_I0, red_chi = esf_fit_simplified(np.mean(ROI,axis=0))  #fit gaussian cdf to data

                
                #p = np.array([p[0], p[1], x_I1, x_I0])
                
                if ((red_chi<=red_chi_tr) and (errs[0]/pars[0] < sigma_err_tr)): #if errs[0]/pars[0] < sigma_err_tr: #0.23 for high int data sets
                    real_space[i] = calculate_esf(ROI, simplified=False,pixel_size=1, disp = 0) 
                    fourier_space[i] = calculate_mtf(ROI, 1, disp=0, med_filt=med_filt,
                                                     fit_to_fourier=fit_to_fourier,
                                                     noise_der=noise_der,
                                                     alpha=alpha)
                else: 
                    real_space[i] = np.nan
                    fourier_space[i] = np.nan
                
                if (
                fourier_space[i][0] == 1
                or fourier_space[i][0] ==0 
                or fourier_space[i][1]>100 
                or fourier_space[i][1]<3
                or (fourier_space[i][2])/(fourier_space[i][0]) > mtf_err_tr #0.13 until 30,15 ; 23 for low int
                or fourier_space[i][2] != fourier_space[i][2] 
                or real_space[i][0]> sigma_tr
                or real_space[i][0] < 0.1 
                ):
                    real_space[i] = np.nan
                    #r[i] = np.nan
                    fourier_space[i][0] = np.nan
                

            except (RuntimeError, ZeroDivisionError, ValueError, RuntimeWarning):
                  
                real_space[i] = np.nan
                fourier_space[i] = np.nan
                
 


        else:
            real_space[i] = np.nan
            fourier_space[i] = np.nan
            

        
    return real_space, fourier_space
    
 
def noise_model_SNR(x,b,c):
    return x/np.sqrt(b*x+c*x**2)

def calc_res_from_model(sigma, SNR):

    roi_len = 50
    x=np.arange(0, roi_len, 0.1)
    lsf = lsf1d(res.esf_p(x, 0.1*sigma , 0.1*x.shape[0]/2, 1, 0))

    freqs = np.arange(-1,1, 2/x.shape[0])
    mtf_th = mtf(lsf)
    mtf_th = mtf_th/np.amax(mtf_th)
    

    mtfparams, mtfcov = curve_fit(gaussian, freqs, mtf_th)
    mtf_err = np.sqrt(np.diag(mtfcov))

    

    
    res_freq = np.abs(mtfparams[0])*np.sqrt(2*np.log(SNR))

    return res_freq
 
def normalize(array):
    return (array - np.min(array))/np.ptp(array)
 
def test_wshed_segm(im, lower = 0.05, upper = 0.08, disp = 0):
    
    el_map = sobel(im)
    
    markers = np.zeros_like(im)
    markers[im < lower] = 1
    markers[im > upper] = 2
    
    segmentation = watershed(el_map, markers)
    
    if disp != 0:
        plt.imshow(segmentation-1)
    
    return segmentation - 1
    
    
def calc_res_dc_from_model(dc, mean_sigma, SNR_model, SNR_model_params, tissue=False):
    
    mean_intensities = np.zeros(dc.shape[0])
    res_freqs = np.zeros(dc.shape[0])
    
    for i in range(dc.shape[0]):
    
        if tissue==False:
            mean_intensities[i] = np.mean(dc[i])
        if tissue==True:
            img_mask = test_wshed_segm(normalize(dc[i]), 0.06, 0.08, 0)
        
        mean_intensities[i] = np.ma.array(dc[i], mask = ~img_mask.astype('bool')).mean()

        SNR = SNR_model(mean_intensities[i], *SNR_model_params)

        res_freqs[i] = calc_res_from_model( sigma=mean_sigma, SNR=SNR)
    

    return mean_intensities, res_freqs
