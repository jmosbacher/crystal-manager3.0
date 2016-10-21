from __future__ import print_function
import numpy as np
import math
import os
from tabulate import tabulate
from scipy.optimize import curve_fit
import scipy.stats as stats
from time import sleep
import pandas as pd


def merge_data_arrays(array1,array2,res=0.065):
    '''
    Needs to be modified to work when one array equals overlap
    :param array1:
    :param array2:
    :param res:
    :return:
    '''
    first_is_left = array1[:, 0].min()<=array2[:, 0].min()
    l,r = {True:(array1,array2),False:(array2,array1)}[first_is_left]  #max(array1[:, 0].min(), array2[:, 0].min())
    overlap_idx1 = np.where((l[:, 0] >= r[:, 0].min()))
    #overlap_idx2 = np.where(np.logical_and((array2[:, 0] >= l), (array2[:, 0] <= r)))
    overlap_cnt = len(overlap_idx1[0])
    l_array, l_overlap = np.split(l,[-overlap_cnt],axis=0)
    r_overlap,r_array = np.split(r, [overlap_cnt], axis=0)
    weights = np.linspace(0,1,num=overlap_cnt)
    overlap = np.empty((overlap_cnt,2))
    overlap[:,0] = (l_overlap[:,0]+r_overlap[:,0])/2.0
    overlap[:,1] = l_overlap[:,1]*weights[::-1] + r_overlap[:, 1]*weights
    final = np.concatenate([x for x in (l_array,overlap,r_array) if len(x)],axis=0)

    return final


def subtract_data_arrays(array1,array2):
    if len(array1)==len(array2):
        result = np.empty_like(array1)
        result[:,0] = (array1[:,0]+array2[:,0])/2.0
        result[:,1] = array1[:,1]-array2[:,1]
        return result
    l = max(array1[:,0].min(),array2[:,0].min())
    r = min(array1[:,0].max(),array2[:,0].max())
    #first_is_left = array1[:, 0].min() <= array2[:, 0].min()
    #l, r = {True: (array1, array2), False: (array2, array1)}[first_is_left]
    overlap1 = array1[np.where(np.logical_and((array1[:,0] >= l),(array1[:,0] <= r)))]
    overlap2 = array2[np.where(np.logical_and((array2[:,0] >= l), (array2[:,0] <= r)))]
    result = np.empty_like(overlap1)

    result[:,0] = np.resize(overlap2[:,0], overlap1[:,0].shape)
    result[:, 1] = np.resize(overlap2[:, 1], overlap1[:, 1].shape)
    result[:, 0] = (overlap1[:, 0] + result[:, 0])/2.0
    result[:, 1] = overlap1[:, 1] - result[:, 1]
    return result



def merge_spectrums(spectrum1,spectrum2,res=0.065):
    """

    :param spectrum1:
    :param spectrum2:
    :return:
    """

    spectrum1.em_wl = (min(spectrum1.em_wl[0],spectrum2.em_wl[0]),max(spectrum1.em_wl[1],spectrum2.em_wl[1]))
    spectrum1.signal = merge_data_arrays(spectrum1.normalized('signal'), spectrum2.normalized('signal'),res=res)
    if len(spectrum2.bg):
        spectrum1.bg = merge_data_arrays(spectrum1.normalized('bg'), spectrum2.normalized('bg'),res=res)
    if len(spectrum2.ref):
        spectrum1.ref = merge_data_arrays(spectrum1.normalized('ref'), spectrum2.normalized('ref'), res=res)
    spectrum1.frames = 1
    spectrum1.exposure = 1.0

    return spectrum1

def merge_experiments(col1,col2):
    """

    :param col1:
    :param col2:
    :return:
    """
    col1.measurements.extend(col2.experiments)
    return col1

def read_ascii_file(path, file_del):
    data = []
    sup = []
    with open(path, 'r') as f:
        sig = True
        for line in f:
            if line in ['\n', '\r\n']:
                sig = False
                continue
            if sig:
                data.append(np.fromstring(line,count=2, sep=file_del))
                #s = line.split(file_del)
                #data.append([eval(s[0]), eval(s[1])])

            else:
                sup.append(line)
    if len(data):
        return np.array(data), sup
    return None



def organize_data(in_data,tags=('sig','bgd','ref'), ext='.asc'):
    '''
    :param data: Dictionary {file name: data array}
    :param tags: text tags to organize by
    :param ext: text file name extension to ignore
    :return: Dictionarey {Name: {signal:array, bg:array, ref:array}
    '''

    out = {}
    for name, data in in_data.items():
        for tag in tags:
            if tag in name:
                if name.replace('_' + tag,'').replace(ext, '') in out.keys():
                    out[name.replace('_' + tag,'').replace(ext, '')][tag] = data
                else:
                    out[name.replace('_' + tag, '').replace(ext, '')] = {tag:data}
    return out

def import_group(path, names,**kwargs):
    delimiter = kwargs.get('delimiter', ' ')
    log = kwargs.get('log', None)
    tool = kwargs.get('tool', None)

    dir_path = path
    data = {}
    for name in names:
        path = os.path.join(dir_path, name)
        if os.path.isfile(path):
            if log is not None:
                print('Reading data from file at:\n %s' % path, file=log)
            result = read_ascii_file(path, delimiter)
            if result is not None:
                data[name] = result
            if log is not None:
                print('File Read.', file=log)
                sleep(0.05)
            elif log is not None:
                print('Skipping (not a file): \n%s' % path, file=log)
    if log is not None:
        print('Organizing Data.', file=log)
    organized = organize_data(data)
    if log is not None:
        print('%d Files successfully imported.' %len(data.keys()), file=log)
        sleep(0.5)
    if tool is not None:
        tool.result = organized
        tool.done = True
    else:
        return organized

def import_folder(path, **kwargs):
    delimiter = kwargs.get('delimiter',' ')
    log = kwargs.get('log',None)
    tool = kwargs.get('tool',None)
    result_name = kwargs.get('result_name',None)
    if not os.path.isdir(path):
        if log is not None:
            print('Cannot find folder:\n%s.' % path, file=log)
        return False

    names = os.listdir(path)
    files = []
    if log is not None:
        print('Searching for files in:\n%s.' %path, file=log)
    for name in names:
        if os.path.isfile(os.path.join(path,name)):
            files.append(name)
    if log is not None:
        print('Files found: %d' %len(files), file=log)
        sleep(0.2)
    #org_data = import_group(path,files,delimiter=delimiter,log=log)
    dir_path = path
    data = {}
    for name in files:
        path = os.path.join(dir_path, name)
        if os.path.isfile(path):
            if log is not None:
                print('Reading data from file at:\n %s' % path, file=log)
            result = read_ascii_file(path, delimiter)
            if result is not None:
                data[name] = result
                if log is not None:
                    print('File read succesfully.', file=log)
                    sleep(0.06)
            elif log is not None:
                print('File import unsuccesfull.', file=log)
        elif log is not None:
            print('Skipping (not a file): \n%s' % path, file=log)
    if log is not None:
        print('Organizing Data.', file=log)
    organized = organize_data(data)
    if log is not None:
        print('Done. Files successfully imported: %d' % len(data.keys()), file=log)
        print('Storing Data...', file=log)
        sleep(0.3)
    if tool is not None:
        #print('Tool is not None' , file=log)
        if result_name is None:
            #print('Result name is None', file=log)
            tool.result = organized
            tool.done = True
        else:
            #print('Result name is not None', file=log)
            tool.result[result_name] = organized
            tool.done = True
    else:
        return organized
    if log is not None:
        print('Finished importing files.', file=log)
        sleep(0.3)
    return True

def bin_data_array(data,nbins=200):
    bins = np.array_split(data,int(nbins),axis=0)
    out = np.empty((int(nbins),2))
    for idx,bin in enumerate(bins):
        out[idx,0] = np.mean(bin[:,0])
        out[idx,1] = np.sum(bin[:,1])
    return out

def data_array_to_text_file(array,path,headers=None,table_fmt='plain',float_fmt='.2e',first_line=None):
    with open(path,'w') as f:
        if first_line is not None:
            print(first_line, file=f)
        print(tabulate(array,headers=headers,tablefmt=table_fmt,floatfmt=float_fmt),file=f)
        return path
    return None

def sum_of_gaussians(N):
    def gaussians(x,p):
        return np.sum([p['a'][i]*np.exp(-(x-p['mean'][i])**2/(2*p['sigma_sqr'][i])) for i in range(N)])

    return gaussians

def fit_sum_of_gaussians(x,f, ranges):
    res = np.mean(np.diff(x))
    y = f / res

    p0 = {'a':[], 'mean':[],'sigma_sqr':[]}

    gaussians = sum_of_gaussians(len(ranges))
    for mn,mx in ranges:
        p0['a'].append(np.where(np.logical_and(y<=mx,y>=mn),y,0.0).max() )
        p0['mean'].append( (mx+mn)/2.0 )
        p0['sigma_sqr'].append( mx-mn )
    try:
        p, pcov = curve_fit(gaussians, x, y, p0=p0)
    except:
        p = None
        pcov = None
    return gaussians,p, pcov

def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta):
    xo = np.float(xo)
    yo = np.float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

def gauss(x, a, mean, sigma):
    return a * np.exp(-(x - mean) ** 2 / (2 * sigma**2))

def gaussian_integral(xmin, xmax, a, mean, sigma, resolution):
    xdata = np.arange(xmin,xmax,resolution)

    return np.sum(gauss(xdata,a,mean,sigma))

def integrate_gaussian(x,f):

    res = np.mean(np.diff(x))
    y = f/res
    a0 = y.max()
    mean0 = np.sum(x * y) / np.sum(y)
    sigma_sqr0 = np.sqrt(np.sum(y * (x - mean0)**2) / np.sum(y))
    try:
        p, pcov = curve_fit(gauss, x, y, p0=[a0, mean0, sigma_sqr0])
    except:
        p = [0.0,0.0,0.0]
    return np.sqrt(2*np.pi*p[2])*p[0]

def pad_with_zeros(data, l,r):
    res = np.mean(np.diff(data[:,0]))
    min_wl = data[:,0].min()
    max_wl = data[:,0].max()
    add_l_wls = np.arange(l,min_wl,res)
    add_l = np.zeros((len(add_l_wls),2))
    add_l[:,0] = add_l_wls
    add_r_wls = np.arange(max_wl+res,r+res, res)
    add_r = np.zeros((len(add_r_wls),2))
    add_r[:, 0] = add_r_wls
    final = np.concatenate((add_l,data,add_r),axis=0)
    return final


def calc_df_statistics(dff, statistic='hist', **kwargs):
    df = dff.fillna(method='ffill')
    if statistic in ['pct_change','diff']:
        return getattr(df,statistic)(**kwargs)
    data = []

    for col_name in df:
        ser = df[col_name]
        if statistic == 'hist':
            bins = kwargs.get('bins', 20)

            cnts, divs = np.histogram(ser, bins=bins)
            data.append(pd.Series(data=divs[:-1] + np.diff(divs) / 2, name=str(col_name) + ' Bin center'))
            data.append(pd.Series(data=cnts,name=str(col_name) + ' Bin count') )


        elif statistic == 'kde':
            nsample = kwargs.get('nsample', 200)
            #ser.dropna(inplace=True)
            arr = ser.as_matrix()
            kern = stats.gaussian_kde(arr)
            rng = np.linspace(arr.min(), arr.max(), nsample)
            data.append(pd.Series(data=rng, name=str(col_name) + ' Counts'))
            data.append(pd.Series(data=kern(rng), name=str(col_name) + ' Value'))
            #data[col_name] = pd.Series(data=kern(rng), index=rng)


    return pd.concat(data,axis=1).fillna(value=0.0)

def wl_to_rgb(wl):
    select = np.select
    # power=np.power
    # transpose=np.transpose
    arange = np.arange

    def factor(wl):
        return select(
            [ wl > 700.,
              wl < 420.,
              True ],
            [ .3+.7*(780.-wl)/(780.-700.),
              .3+.7*(wl-380.)/(420.-380.),
              1.0 ] )

    def raw_r(wl):
        return select(
            [ wl >= 580.,
              wl >= 510.,
              wl >= 440.,
              wl >= 380.,
              True ],
            [ 1.0,
              (wl-510.)/(580.-510.),
              0.0,
              (wl-440.)/(380.-440.),
              0.0 ] )

    def raw_g(wl):
        return select(
            [ wl >= 645.,
              wl >= 580.,
              wl >= 490.,
              wl >= 440.,
              True ],
            [ 0.0,
              (wl-645.)/(580.-645.),
              1.0,
              (wl-440.)/(490.-440.),
              0.0 ] )

    def raw_b(wl):
        return select(
            [ wl >= 510.,
              wl >= 490.,
              wl >= 380.,
              True ],
            [ 0.0,
              (wl-510.)/(490.-510.),
              1.0,
              0.0 ] )

    gamma = 0.80
    def correct_r(wl):
        return np.round(math.pow(factor(wl)*raw_r(wl),gamma), 2)
    def correct_g(wl):
        return np.round(math.pow(factor(wl)*raw_g(wl),gamma), 2)
    def correct_b(wl):
        return np.round(math.pow(factor(wl)*raw_b(wl),gamma), 2)


    return (correct_r(wl),correct_g(wl),correct_b(wl))

def color_map(wl):
    # ['r', 'g', 'b', 'y', 'g', 'k', 'm', 'c', 'k']
    col = 'b'
    if wl >485:
        col = 'c'
    if wl > 500:
        col = 'g'
    if wl > 565:
        col ='y'
    if wl > 590:
        col = 'm'
    if wl > 625:
        col = 'r'
    return col


