import pandas as pd
from hashlib import md5
import pickle
import pystan
import os
import yfinance as yahoo
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


tickers = ["MMM", "ABT", "ABBV", "ABMD", "ACN", "ATVI", "ADBE", "AMD", "AAP", "AES", "AMG", "AFL", "A", "APD", "AKAM", "ALK", "ALB", "ARE", "ALXN", "ALGN", "ALLE", "AGN", "ADS", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP", "AXP", "AIG", "AMT", "AWK", "AMP", "ABC", "AME", "AMGN", "APH", "ADI", "ANSS", "ANTM", "AON", "AOS", "APA", "AIV", "AAPL", "AMAT", "APTV", "ADM", "ARNC", "ANET", "AJG", "AIZ", "T", "ADSK", "ADP", "AZO", "AVB", "AVY", "BHGE", "BLL", "BAC", "BK", "BAX", "BBT", "BDX", "BRK-B", "BBY", "BIIB", "BLK", "HRB", "BA", "BWA", "BXP", "BSX", "BMY", "AVGO", "BR", "BF-B", "CHRW", "COG", "CDNS", "CPB", "COF", "CPRI", "CAH", "KMX", "CCL", "CAT", "CBOE", "CBRE", "CBS", "CDW", "CE", "CELG", "CNC", "CNP", "CTL", "CERN", "CF", "SCHW", "CHTR", "CVX", "CMG", "CMI", "CVS", "DHI", "DHR", "DRI", "DVA", "DE", "DAL", "XRAY", "DVN", "FANG", "DLR", "DFS", "DISCA", "DISCK", "DISH", "DG", "DLTR", "D", "DOV", "DOW", "DTE", "DUK", "DRE", "DD", "DXC", "ETFC", "EMN", "ETN", "FLT", "FLIR", "FLS", "FMC", "F", "FTNT", "FTV", "FBHS", "BEN", "FCX", "GPS", "GRMN", "IT", "GD", "GE", "GIS", "GM", "GPC", "GILD", "GL", "GPN", "GS", "GWW", "HAL", "HBI", "HOG", "HIG", "HAS", "HCA", "HCP", "HP", "HSIC", "HSY", "HES", "HPE", "HLT", "HFC", "HOLX", "HD", "HON", "HRL", "HST", "HPQ", "HUM", "HBAN", "IDXX", "INFO", "ITW", "ILMN", "IR", "INTC", "ICE", "IBM", "INCY", "IP", "IPG", "IFF", "INTU", "ISRG", "IVZ", "IPGP", "IQV", "IRM", "JKHY", "JEC", "JBHT", "SJM", "JNJ", "JCI", "JPM", "JNPR", "KSU", "K", "KEY", "KEYS", "KMB", "KIM", "KMI", "KLAC", "KSS", "KHC", "KR", "LB", "LHX", "LH", "LRCX", "LW", "LVS", "LEG", "LDOS", "LEN", "LLY", "LNC", "LIN", "LKQ", "LMT", "L", "LOW", "LYB", "MTB", "MAC", "M", "MRO", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", "MA", "MKC", "MXIM", "MCD", "MCK", "MDT", "MRK", "MET", "MTD", "MGM", "MCHP", "MU", "MSFT", "MAA", "MHK", "TAP", "MDLZ", "MNST", "MCO", "MS", "MOS", "MSI", "MYL", "NDAQ", "NOV", "NTAP", "NFLX", "NWL", "NEM", "NWSA", "NWS", "NEE", "NLSN", "NKE", "NI", "NBL", "JWN", "NSC", "NTRS", "NOC", "NCLH", "NRG", "NUE", "NVDA", "ORLY", "OXY", "OMC", "OKE", "ORCL", "PCAR", "PKG", "PH", "PAYX", "PYPL", "PNR", "PBCT", "PEP", "PKI", "PRGO", "PFE", "PM", "PSX", "PNW", "PXD", "PNC", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG", "PSA", "PHM", "PVH", "QRVO", "PWR", "QCOM", "DGX", "RL", "RJF", "RTN", "O", "REG", "REGN", "RF", "RSG", "RMD", "RHI", "ROK", "ROL", "ROP", "ROST", "RCL", "CRM", "SBAC", "SLB", "STX", "SEE", "SRE", "SHW", "SPG", "SWKS", "SLG", "SNA", "SO", "LUV", "SPGI", "SWK", "SBUX", "STT", "SYK", "STI", "SIVB", "SYMC", "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO", "TPR", "TGT", "TEL", "FTI", "TFX", "TXN", "TXT", "TMO", "TIF", "TWTR", "TJX", "TSCO", "TDG", "TRV", "TRIP", "TSN", "UDR", "ULTA", "USB", "UAA", "UA", "UNP", "UAL", "UNH", "UPS", "URI", "UTX", "UHS", "UNM", "VFC", "VLO", "VAR", "VTR", "VRSN", "VRSK", "VZ", "VRTX", "VIAB", "V", "VNO", "VMC", "WAB", "WMT", "WBA", "DIS", "WM", "WAT", "WEC", "WCG", "WFC", "WELL", "WDC", "WU", "WRK", "WY", "WHR", "WMB", "WLTW", "WYNN", "XEL", "XRX", "XLNX", "XYL", "YUM", "ZBH", "ZION", "ZTS"]


def data_loader(begin, finish, output_file): 
    data_r = pd.DataFrame()
    df_r = pd.DataFrame()
    returns_r = pd.DataFrame(columns=tickers)
    loaded_r = pd.DataFrame()
    df_l_r = []
    for ticker in tickers:
        loaded_r = yahoo.download(ticker, start=begin, end=finish).loc[:,['Adj Close']]
        df_r = loaded_r.pct_change(fill_method='ffill')
        df_l_r.append(df_r)
    df_r = pd.concat(df_l_r, axis=1)
    df_r.columns = tickers
    df_r.iloc[1:, :]
    df_r.to_csv(output_file)

def delete_nans(input_file, output_file):
    dirty_data = pd.read_csv(input_file, index_col='Date', parse_dates=['Date']).fillna(0)
    data = pd.DataFrame(dirty_data)
    data.to_csv(output_file)
    #saves csv like dataframe with header and index 
    
def get_returns(input_file, N):
    returns = pd.read_csv(input_file, index_col='Date', parse_dates=['Date']).iloc[2:, :]
    returns = returns.T.sample(N).T
    D = returns.shape[0]
    print('Number of columns with only NaNs: {}'.format(sum(returns.isna().sum(axis=0) == D)))
    print('Number of lines with only NaNs: {}'.format(sum(returns.isna().sum(axis=1) == D)))
    print('Number of NaNs: {}'.format(returns.isna().sum().sum()))
    returns = pd.DataFrame.dropna(returns, axis='columns', how='all')
    print('shape Dataframe: {}'.format(returns.shape)) 
    return returns.fillna(value=0.)

def plot_example_returns(input_file, number_stocks):
    data_use = pd.read_csv(input_file, index_col='Date', parse_dates=['Date']).iloc[:,:number_stocks]
    stock_list = data_use.columns
    display(data_use.head())
    if number_stocks <= 40:
        fig = plt.figure(figsize=(40,20))
        
        ax = fig.add_subplot(121)
        np.cumprod(1+data_use, axis=0).plot(ax=ax, title='stock price')
        np.cumprod(1+data_use.mean(axis=1)).plot(ax=ax, label='mean', color='black')
        ax.legend()
        
        ax = fig.add_subplot(122)
        data_use.plot(ax=ax, title='returns')
        ax.legend()
        plt.show()
        
def get_data_subsets(df,dur):
    df = np.array(df)
    leng = df.shape[0]
    print (df.shape[0])
    x=[]
    y=[]
    for i in range(leng):
        if dur+i>=leng:
            print (i)
            break
        x.append(df[i:dur+i,:])
        y.append(df[dur+i,:])
    return np.array(x),np.array(y)

def vb(data_dict, stan_model, init='random', iter=10000, tries=5, num=0):
    """
        vb - variational bayes
        Approximates the posterior with independent gaussians \
        and returns samples from the gaussians. 
    """
    try:
        fit = stan_model.vb(data=data_dict, diagnostic_file='d_{}.csv'.format(num),
                       sample_file='s_{}.csv'.format(num), elbo_samples=100, init=init,
                       iter=iter)
        diagnostic = pd.read_csv('d_{}.csv'.format(num), 
                                 names=['iter', 'time_in_seconds', 'ELBO'], 
                                 comment='#', sep=',')
        sample = pd.read_csv('s_{}.csv'.format(num), comment='#', sep=',')
        #print('vb - ELBO: {}'.format(diagnostic.loc[:,'ELBO'].values[-1]))
        os.remove('d_{}.csv'.format(num))
        os.remove('s_{}.csv'.format(num))
    except pd.errors.ParserError:
        print('pandas ParserError - trying again.')
        diagnostic, sample = vb(data_dict, stan_model, init=init, 
                                iter=iter, tries=tries, num=num)
        
    for _ in range(tries-1):
        diagnostic_, sample_ = vb(data_dict, stan_model, init, 
                                  iter=iter, tries=1, num=num)
        if diagnostic.loc[:,'ELBO'].values[-1] < diagnostic_.loc[:,'ELBO'].values[-1]:
            diagnostic = diagnostic_
            sample = sample_
        del(diagnostic_, sample_)
    return diagnostic, sample


def quick_comp_plot(pred_comp):
        #Visualisation for fast quantification of result quality
        fig = plt.figure(figsize=(40,20))
        
        ax = fig.add_subplot(331)
        ax = sns.regplot(x=pred_comp.iloc[0,:], y=pred_comp.iloc[20,:] )
        ax = sns.lineplot(x=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[0,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[0,:]), num=50),
                  y=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[0,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[0,:]), num=50))
        
        ax = fig.add_subplot(332)
        ax = sns.regplot(x=pred_comp.iloc[1,:], y=pred_comp.iloc[21,:] )
        ax = sns.lineplot(x=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[1,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[1,:]), num=50),
                  y=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[1,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[1,:]), num=50))
        
        ax = fig.add_subplot(333)
        ax = sns.regplot(x=pred_comp.iloc[2,:], y=pred_comp.iloc[22,:] )
        ax = sns.lineplot(x=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[2,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[2,:]), num=50),
                  y=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[2,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[2,:]), num=50))
        
        ax = fig.add_subplot(334)
        ax = sns.regplot(x=pred_comp.iloc[3,:], y=pred_comp.iloc[23,:] )
        ax = sns.lineplot(x=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[3,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[3,:]), num=50),
                  y=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[3,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[3,:]), num=50))
        
        ax = fig.add_subplot(335)
        ax = sns.regplot(x=pred_comp.iloc[4,:], y=pred_comp.iloc[24,:] )
        ax = sns.lineplot(x=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[4,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[4,:]), num=50),
                  y=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[4,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[4,:]), num=50))
        
        ax = fig.add_subplot(336)
        ax = sns.regplot(x=pred_comp.iloc[5,:], y=pred_comp.iloc[25,:] )
        ax = sns.lineplot(x=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[5,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[5,:]), num=50),
                  y=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[5,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[5,:]), num=50))
        
        ax = fig.add_subplot(337)
        ax = sns.regplot(x=pred_comp.iloc[6,:], y=pred_comp.iloc[26,:] )
        ax = sns.lineplot(x=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[6,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[6,:]), num=50),
                  y=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[6,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[6,:]), num=50))
        
        ax = fig.add_subplot(338)
        ax = sns.regplot(x=pred_comp.iloc[7,:], y=pred_comp.iloc[27,:] )
        ax = sns.lineplot(x=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[7,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[7,:]), num=50),
                  y=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[7,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[7,:]), num=50))
        
        ax = fig.add_subplot(339)
        ax = sns.regplot(x=pred_comp.iloc[8,:], y=pred_comp.iloc[28,:] )
        ax = sns.lineplot(x=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[8,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[8,:]), num=50),
                  y=np.linspace(1.1*pd.DataFrame.min(pred_comp.iloc[8,:]),
                                1.1*pd.DataFrame.max(pred_comp.iloc[8,:]), num=50))
        plt.savefig('plot.png')
        plt.plot()


#from pyStan website, pickles compiled stan model into cache.
def StanModel_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm 

