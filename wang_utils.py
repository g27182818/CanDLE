import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

# Expression threshold for a gene to be considered turned off
threshold = 5

# Declare filter used to remove genes that are turned off
filter_fun = lambda mydata, thr, pat: np.apply_along_axis(lambda x: np.sum(x > thr) >= pat, 0, mydata.X)

def GetFilterCutoff(mydata, turnedOffGene):
    
    # Get global threshold
    global threshold

    # Define vector of number of patients that should be over the threshold (threshold in Expression)
    steps = np.arange(5, 605, 5)

    # Loop through the number of patients
    for i in steps:

        # print(f'Evaluating {i} patients')
        
        # Get boolean index of genes with more than threshold (in expression space) in at least i patients
        filter = filter_fun(mydata, threshold, i)

        # Get the boolean intersection of valid genes and genes that are turned off
        filter = filter & turnedOffGene

        # Filter the data and log2(x+1) transform it
        filtered = mydata[:, filter].copy()
        filtered.X = np.log2(filtered.X + 1)

        # Try to run ComBat on the filtered data
        try:
            combat_edata1 = sc.pp.combat(filtered, key='is_tcga', covariates=['healthy'])
            print("ComBat Success")
        
        except:
            print(f"ComBat Failure, {filter.sum()} genes left")
        
        else:
            for j in range(i-4, i):

                # print(f'Evaluating {j} patients')

                # Get boolean index of genes with more than threshold (in expression space) in at least j patients
                filter = filter_fun(mydata, threshold, j)

                # Get the boolean intersection of valid genes and genes that are turned off
                filter = filter & turnedOffGene

                # Filter the data and log2(x+1) transform it
                filtered = mydata[:, filter].copy()
                filtered.X = np.log2(filtered.X + 1)

                # Try to run ComBat on the filtered data
                try:
                    combat_edata2 = sc.pp.combat(filtered, key='is_tcga', covariates=['healthy'])
                    print("ComBat Success")
                except:
                    print(f"ComBat Failure, {filter.sum()} genes left")
                else:
                    return j
            return i
    return 0


def GetTroubleGene(mydata, filterCutoff, turnedOffGene):

    # Get global threshold
    global threshold

    # Find genes that pass with the parameter filterCutoff
    filterPass = filter_fun(mydata, threshold, filterCutoff)
    filterPass = filterPass & turnedOffGene

    # Find genes that pass the filter with one less patient (combat fails)
    filterFail = filter_fun(mydata, threshold, filterCutoff - 1)
    filterFail = filterFail & turnedOffGene

    # Find the difference between the two filters
    diff = np.logical_xor(filterPass, filterFail)
    dub_genes = np.where(diff)[0]

    good_genes = []
    for i in dub_genes:
        print(i)
        filter = filterPass.copy()
        filter[i] = True
        filtered = np.log2(mydata[filter, :] + 1)

        try:
            sc.pp.combat(filtered, key='is_tcga', covariates=['healthy'])
            good_genes.append(i)
        except:
            pass

    bad_genes = np.setdiff1d(dub_genes, good_genes)
    return bad_genes


def ComBatWrapper(mydata, iter_n):

    # Get global threshold
    global threshold

    # Find cutoff that makes ComBat work
    turnedOffGene = pd.Series(data=True, index=mydata.var_names, dtype=bool)
    filterCutoff = GetFilterCutoff(mydata, turnedOffGene)
    print(f'Cutoff to filter genes = {filterCutoff}')


    for i in range(1, iter_n+1):
        print(f'Iteration {i}')
        if filterCutoff <= 2:
            filter = filter_fun(mydata, threshold, filterCutoff)
            filter[~turnedOffGene] = False
            
            filtered = mydata[:, filter].copy()
            filtered.X = np.log2(filtered.X + 1)
            
            try:
                edata = sc.pp.combat(filtered, key='is_tcga', covariates=['healthy'], inplace=False)
                return edata
            except:
                print("ComBat Failure")
                return None
        else:
            filteredGene = GetTroubleGene(mydata, filterCutoff, turnedOffGene)
            turnedOffGene[filteredGene] = False
            
            # Find new cutoff
            newfilterCutoff = GetFilterCutoff(mydata, turnedOffGene)
            if newfilterCutoff != filterCutoff:
                filterCutoff = newfilterCutoff
                print(f'New cutoff to filter genes = {filterCutoff}')
            else:
                filter = filter_fun(mydata, threshold, filterCutoff)
                filter[~turnedOffGene] = False
                filtered = np.log2(mydata.loc[filter, :].add(1))
                edata = sc.pp.combat(filtered, key='is_tcga', covariates=['healthy'])
                return edata

    filter = filter_fun(mydata, threshold, filterCutoff)
    filter[~turnedOffGene] = False
    filtered = np.log2(mydata.loc[filter, :].add(1))
    edata = sc.pp.combat(filtered, key='is_tcga', covariates=['healthy'], inplace=False)
    return edata