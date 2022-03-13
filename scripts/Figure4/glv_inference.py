import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.interpolate import CubicSpline
from copy import deepcopy
import numpy as np
import arviz as az
import scipy.stats.kde as kde
from sklearn.preprocessing import StandardScaler

###############################
# Function: compute d/dt log(X)
###############################
def compute_ddt_log_phi(
    df_meta,                  # dataframe: rows are sample_id, columns inlcude subject_id, time_point, group, strength
    df_abun,                  # dataframe: rows are sample_id, columns are taxa, values are relative or absolute abundance
    taxa2include=None,        # list: taxa to be simulated in the gLV model
    numtaxa2include=20,       # int: number of taxa to be simulated in the gLV model (this option is only active when taxa2include is None)
    method='spline',          # str: available methods include spline and gradient (central difference)
    log_transform=True        # boolean: set log_transform to true to use gLV
):
    # make sure that sample ids are unique in both meta data and abundance table
    sample_ids_from_meta = list(df_meta.index)
    if len(sample_ids_from_meta) != len(set(sample_ids_from_meta)):
        raise RuntimeError("duplicate sample ids in meta data.")
    sample_ids_from_abun = list(df_abun.index)
    if len(sample_ids_from_abun) != len(set(sample_ids_from_abun)):
        raise RuntimeError("duplicate sample ids in otu table.")

    # if series_id does not exist, assume each subject has a unique time series
    df_meta2 = deepcopy(df_meta)
    if "series_id" not in list(df_meta.columns):
        df_meta2["series_id"] = df_meta2["subject_id"]

    # remove time series that have a single sample (at least two time points are required)
    time_series_to_remove = []
    for sid in set(df_meta2.series_id):
        if len(df_meta2[df_meta2.series_id==sid]) == 1:
            time_series_to_remove.append(sid)
    df_meta2 = df_meta2[~df_meta2.series_id.isin(time_series_to_remove)]
    if len(df_meta2) == 0:
        print("None of the time series have at least two samples.")
        return None

    # if strength is not specified, it is assigned to 1.0 for all groups
    if 'strength' not in list(df_meta2.columns):
        df_meta2['strength'] = [1.0]*len(df_meta2)

    # select taxa to be simulated in the gLV model
    df_abun2 = deepcopy(df_abun)
    if taxa2include is None:
        if numtaxa2include < 1:
            raise RuntimeError("numtaxa2include must be at least 1. current value = %d"%(numtaxa2include))
        else:
            df_abun2_T = df_abun2.T
            df_abun2_T['mean'] = df_abun2_T.mean(axis=1)
            df_abun2_T = df_abun2_T.sort_values(by=['mean'],axis=0,ascending=False)
            df_abun2_T = df_abun2_T.drop('mean', axis=1)
            df_abun2 = df_abun2_T.iloc[0:numtaxa2include].T
    else:
        if len(taxa2include) == 0:
            raise RuntimeError("taxa2include must contain at least 1 taxon. current length = %d"%(len(taxa2include)))
        df_abun2 = df_abun2[[taxaid for taxaid in df_abun2.columns if taxaid in taxa2include]]
    num_taxa_included = len(df_abun2.columns)

    # normalize abundance
    df_abun2 = df_abun2/df_abun2.max().max() # normalize abundance (maximum -> 1)

    # join meta data and abundance file
    df_join = pd.merge(df_meta2, df_abun2, left_index=True, right_index=True, how='inner')

    # calculate log-derivatives of relative abundance
    df_output = None
    for sid in set(df_join.series_id):
        df_y = df_join[df_join.series_id==sid].sort_values(by='time_point').drop(['series_id','subject_id','group','strength'], axis=1)
        df_y = df_y.groupby(df_y.time_point).agg(np.mean) # in case there are multiple samples at the same timepoint

        # to apply log, replace 0 with minimum fraction within each sample
        df_y_T = df_y.T
        for tps in df_y_T.columns:
            df_y_T.loc[df_y_T[tps]==0, tps] = df_y_T.loc[df_y_T[tps]>0, tps].min()
        df_y = df_y_T.T
        assert np.count_nonzero(df_y) == df_y.shape[0]*df_y.shape[1]

        if log_transform:
            df_logy = np.log(df_y)
        else:
            df_logy = df_y

        # make sure that x is increasing in time points
        xdata = list(df_y.index)
        if not (pd.Series(xdata).is_unique and pd.Series(xdata).is_monotonic_increasing):
            raise RuntimeError("time points must strictly monotonically increase.")

        if len(xdata)>1: # compute derivative needs at least two data points

            # calculate forward quotient
            df_dlogydt = pd.DataFrame(index=xdata)
            for taxon in df_y.columns:
                if method.lower() == 'spline':
                    cs = CubicSpline(xdata, list(df_logy[taxon]))
                    df_dlogydt[taxon] = cs(xdata, 1)
                elif method.lower() == 'central':
                    df_dlogydt[taxon] = np.gradient(list(df_logy[taxon]), xdata)
                else:
                    # by default: spline
                    cs = CubicSpline(xdata, list(df_logy[taxon]))
                    df_dlogydt[taxon] = cs(xdata, 1)

            # convert from wide format to long format
            df_dlogydt = df_dlogydt.stack().reset_index()
            df_dlogydt.columns=['time_point','taxa','ddt_log_phi']
            df_y = df_y.stack().reset_index()
            df_y.columns=['time_point','taxa','phi']

            # combine the two tables
            df_tmp = pd.merge(df_dlogydt, df_y, left_on=['time_point','taxa'], right_on=['time_point','taxa'], how='inner')

            # append subject information
            df_tmp['series_id'] = sid

            if df_output is None:
                df_output = deepcopy(df_tmp)
            else:
                df_output = pd.concat([df_output, df_tmp], axis=0)

    # reorder columns
    df_output = df_output[['series_id','time_point','taxa','phi','ddt_log_phi']]

    # add sample ids and perturbation strength
    # if multiple samples are collected on the same timepoint, they will be conum_taxaected by '_and_'
    df_output = pd.merge(
        df_output,
        df_meta2.reset_index().groupby(['series_id','subject_id','time_point','group','strength'])['sample_id'].apply(lambda x: '_and_'.join(x)).reset_index(),
        left_on=['series_id','time_point'],
        right_on=['series_id','time_point'],
        how='inner'
    )
    # make sure all sample ids are unique
    assert len(df_output.sample_id) == len(set(df_output.sample_id))*num_taxa_included

    df_output = df_output.sort_values(['subject_id','series_id','time_point'])
    return df_output

#######################################
# Function: Generate regression matrics
#######################################
def generate_XY_matrics(
    df_input,               # dataframe: columns are 'group','subject_id','time_point','taxa','phi','ddt_log_phi','sample_id'
    reference_group=None,   # str: reference group
    standardize_y=False     # boolean: whether to standardize Y matrix
):
    # create Y matrix
    df_logderiv = pd.pivot_table(df_input[['sample_id','taxa','ddt_log_phi']], index='sample_id', columns='taxa', values='ddt_log_phi')
    assert df_logderiv.isna().sum().sum()==0 # make sure no missing values
    simulated_samples = list(df_logderiv.index)
    num_samples = len(simulated_samples)
    simulated_taxa = list(df_logderiv.columns)
    num_taxa = len(simulated_taxa)

    Ymat = df_logderiv.values
    Ymat = Ymat.flatten(order='F')
    if standardize_y:
        Ymat = StandardScaler().fit_transform(Ymat.reshape(-1,1)).reshape(1,-1)[0] # standardize

    # create X matrix
    df_group = pd.pivot_table(df_input[['sample_id','group','strength']], index='sample_id', columns='group', values='strength').fillna(0.0)
    df_group = df_group.loc[simulated_samples] # keep the same order of samples
    if reference_group is not None:
        df_group = df_group.drop(reference_group, axis=1) # drop reference group
    simulated_groups = list(df_group.columns)
    num_groups = len(simulated_groups)

    df_y = pd.pivot_table(df_input[['sample_id','taxa','phi']], index='sample_id', columns='taxa', values='phi')
    df_y = df_y.loc[simulated_samples, simulated_taxa] # keep the same order of samples and taxa
    assert df_y.isna().sum().sum()==0 # make sure no missing values

    Xmat = np.zeros(shape=(num_taxa*num_samples, (1+num_taxa+num_groups)*num_taxa))
    for k in np.arange(num_taxa):
        start_index = k*(1+num_taxa+num_groups)
        Xmat[k*num_samples:(k+1)*num_samples,start_index] = 1.0 # basal growth rate
        Xmat[k*num_samples:(k+1)*num_samples,start_index+1:start_index+1+num_taxa] = df_y.values # pairwise interactions
        Xmat[k*num_samples:(k+1)*num_samples,start_index+1+num_taxa:start_index+1+num_taxa+num_groups] = df_group.values # perturbations

    return Xmat, Ymat, simulated_samples, simulated_taxa, simulated_groups

#############################################
# Function: Generate input files for CMD stan
#############################################
def write_stan_input_file(
    prefix,                     # str: prefix of file names
    stan_path_dir,              # str: directory where stan input files are directed
    Xmat,                       # nd array: X matrix
    Ymat,                       # nd array: Y matrix
    simulated_taxa,             # list: taxa ids
    simulated_groups,           # list: perturbations
    sigma_ub=1,                 # float: upper bound for sigma
    normal_prior_std=1,         # float: normal prior std for alpha, beta and epsilon
    neg_self_int=False          # boolean: whether to implement negative constraint on self-self interactions
):
    # number of taxa
    num_taxa = len(simulated_taxa)
    num_groups = len(simulated_groups)

    # write data to stan program files
    json_str = '{\n"N" : %d,\n'%(len(Ymat))
    json_str += '\"dlogX\" : [%s],\n'%(','.join(list(Ymat.astype(str))))
    for k1,c1 in enumerate(simulated_taxa):
        start_index = k1*(1+num_taxa+num_groups)
        # basal growth rate
        json_str += '\"growth_%s\" : [%s],\n'%(c1,','.join(list(Xmat[:,start_index].astype(str))))
        # pairwise interactions
        for k2,c2 in enumerate(simulated_taxa):
            v = list(Xmat[:,start_index+1+k2].astype(str))
            json_str += '\"interaction_%s_%s\" : [%s],\n'%(c1,c2,','.join(v))
        # response to perturbations
        for k2,c2 in enumerate(simulated_groups):
            json_str += '\"perturbation_%s_%s\" : [%s],\n'%(c1,c2,','.join(list(Xmat[:,start_index+1+num_taxa+k2].astype(str))))
    json_str = json_str[:-2] + '}'
    text_file = open("%s/%s.data.json"%(stan_path_dir, prefix), "w")
    text_file.write("%s" % json_str)
    text_file.close()

    # write stan program
    # data block
    model_str = 'data {\n'
    model_str += '\tint<lower=0> N;\n'
    model_str += '\tvector[N] dlogX;\n'
    for c1 in simulated_taxa:
        model_str += '\tvector[N] growth_%s;\n'%(c1)
        for c2 in simulated_taxa:
            model_str += '\tvector[N] interaction_%s_%s;\n'%(c1,c2)
        for c2 in simulated_groups:
            model_str += '\tvector[N] perturbation_%s_%s;\n'%(c1,c2)
    model_str += '}\n'

    # parameter block
    model_str += 'parameters {\n\treal<lower=0,upper=%2.2f> sigma;\n'%(sigma_ub)
    for c1 in simulated_taxa:
        model_str += '\treal alpha_%s;\n'%(c1) # basal growth rate
        for c2 in simulated_taxa:
            if c1 != c2:
                model_str += '\treal beta_%s_%s;\n'%(c1,c2) # pairwise interactions
            else:
                if neg_self_int == True:
                    model_str += '\treal<upper=0> beta_%s_%s;\n'%(c1,c2)
                else:
                    model_str += '\treal beta_%s_%s;\n'%(c1,c2)
        for c2 in simulated_groups:
            model_str += '\treal epsilon_%s_%s;\n'%(c1,c2) # response to perturbations
    model_str += '}\n'

    # model block
    # all gLV model parameters have a normal prior with mean 0 and standard deviation
    model_str += 'model {\n\tsigma ~ uniform(0,%2.2f);\n'%(sigma_ub)
    for c1 in simulated_taxa:
        model_str += '\talpha_%s ~ normal(0,%2.2f);\n'%(c1,normal_prior_std) # growth rate
        for c2 in simulated_taxa:
            model_str += '\tbeta_%s_%s ~ normal(0,%2.2f);\n'%(c1,c2,normal_prior_std) # pairwise interactions
        for c2 in simulated_groups:
            model_str += '\tepsilon_%s_%s ~ normal(0,%2.2f);\n'%(c1,c2,normal_prior_std) # resonse to perturbations
    model_str += '\tdlogX ~ normal('
    for c1 in simulated_taxa:
        model_str += 'alpha_%s*growth_%s+'%(c1,c1) # growth rate
        for c2 in simulated_taxa:
            model_str += 'beta_%s_%s*interaction_%s_%s+'%(c1,c2,c1,c2) # pairwise interactions
        for c2 in simulated_groups:
            model_str += 'epsilon_%s_%s*perturbation_%s_%s+'%(c1,c2,c1,c2) # resonse to perturbations
    model_str = model_str[:-1] + ', sigma);\n}'
    text_file = open("%s/%s.stan"%(stan_path_dir, prefix), "w")
    text_file.write("%s" % model_str)
    text_file.close()

    return

##################################
# Function: Parse CMD stan output
##################################
def parse_stan_output(
    stan_output_path,       # list: path to output stan input files
    simulated_taxa,         # list: taxa ids
    simulated_groups,       # list: perturbations
    sig_only=False          # bool: show significant coefficients only
):
    n_files = len(stan_output_path)
    fit = az.from_cmdstan(stan_output_path)

    # process posterior distribution of each variable
    res = []
    for taxa1 in simulated_taxa:
        all_vars = []
        all_vars.append(['alpha_%s'%(taxa1), taxa1, 'growth'])
        for taxa2 in simulated_taxa:
            all_vars.append(['beta_%s_%s'%(taxa1,taxa2), taxa1, taxa2])
        for group in simulated_groups:
            all_vars.append(['epsilon_%s_%s'%(taxa1,group), taxa1, group])
        for var in all_vars:
            data = []
            for i in np.arange(0,n_files):
                data.extend(list(fit.posterior[var[0]][i].values))
            x0,x1 = az.hdi(np.array(data), hdi_prob=0.95)
            res.append([var[1], var[2], np.mean(data), np.std(data), x0, x1, x0*x1>0])
    df_parsed = pd.DataFrame(res, columns = ['Var1','Var2','Mean','STD','95CI_left','95CI_right','Sig'])

    if sig_only==True:
        return df_parsed[df_parsed.Sig==True]
    else:
        return df_parsed
