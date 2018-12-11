import pandas as pd
import numpy as np




def poisson_deviance(y, p):

    d = -2 * np.where(y == 0, -(y - p), (y * np.log(y / p)) - (y - p))

    deviance = sum(d)

    return(deviance)




def gamma_deviance(y, p, w):

    d = -2 * w * (-np.log(y / p) + ((y - p)/p))

    deviance = sum(d)

    return(deviance)






def proportion_deviance_explained(y, p, w, family):

    assert family in ['poisson', 'gamma'], 'family must be poisson or gamma'

    if family == 'poisson':

        deviance = poisson_deviance(y, p)

        null_deviance = poisson_deviance(y, w * np.sum(y) / np.sum(w))

    elif family == 'gamma':

        deviance = gamma_deviance(y, p, w)

        null_deviance = gamma_deviance(y, np.sum(y * w) / np.sum(w), w)

    propn_deviance_explained = (null_deviance - deviance) / null_deviance

    return(deviance, null_deviance, propn_deviance_explained)


