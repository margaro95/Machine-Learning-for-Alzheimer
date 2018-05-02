def get_ensemble_ratio():
    from numpy import load, median, nditer, array, ravel
    from ensemble_DCLvsControl import ensemble_build
    import pdb
    ls_ensemble_ratios = []
    ensembles = 300
    #pdb.set_trace()
    y_score_matrix = ensemble_build(ensembles)
    classification = []

    for g in nditer(y_score_matrix):
        if g > 0.5:
            classification.append(1)
        else:
            classification.append(0)

    classification = array(classification)
    classification = classification.reshape((59, ensembles))
    #pdb.set_trace()
    for nensemble in range(ensembles):
        final_classification = median(classification[:, :(nensemble+1)], axis=1)
        tn = (30 - sum(final_classification[:30]))
        tp = sum(final_classification[30:59])
        ls_ensemble_ratios.append((tp+tn)/59)
    return(ls_ensemble_ratios)
