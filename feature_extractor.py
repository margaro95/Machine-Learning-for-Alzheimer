import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from nilearn.connectome import ConnectivityMeasure as CnMsre
import time


def _load_PLV(band):
    """Load each PLV matrix.
    Take notice that fmri_filename is a pd.Series containing a column of PLV
    array belonging to the same band.
    """
    data = []
    for subject_array in band:
        data.append(subject_filename[0])
    return data


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # make a transformer which will load the time series and compute the
        # connectome matrix
        # no sé por qué no usa el basc197
        self.band_names = ["Delta",
                           "Theta",
                           "Alpha",
                           "Beta1",
                           "Beta2",
                           "Beta",
                           "Gamma",
                           "HighGamma1",
                           "HighGamma2"
                           ]
        # Se transforma el pipeline del starting kit en un diccionario de
        # pipelines con keys iguales a fmri_filenames excepto motions
        self.fmri_transformers = {col: make_pipeline(FcnTrans(func=_load_fmri,
                                                     validate=False), StandardScaler())
                                  for col in self.fmri_names}

        self.pca = PCA(n_components=0.99)

    def fit(self, X_df, y):
        # Esto es lo que no pudiste hacer cuando querías coger todas las
        # features.
        # Debería haberlo hecho así, más limpio
        # fmri_filenames = {col: X_df[col]
        #                   for col in self.fmri_names}
        # A no ser que el fmri_names tenga más elementos de los que tiene
        # X_df.columns
        # Atento, fenómeno, que está utilizando un list comprehension en un
        # diccionario       TRUCAZO
        fmri_filenames = {col: X_df[col]
                          for col in X_df.columns if col in self.fmri_names}

        for fmri in self.fmri_names:
            # Que aparezca este if aquí me hace pensar que hay algún elemento
            # en fmri_names que no está en las columnas del dataset
            if fmri in fmri_filenames.keys():
                # Ese end en el print hace que los prints se vayan acumulando
                # en horizontal en vez de en vertical. Lo quiere hacer para
                # coger después y agregarle el tiempo que ha tardado justo al
                # lado.          TRUCAZO
                print("Fitting", fmri, end="")
                start = time.time()
                self.fmri_transformers[fmri].fit(fmri_filenames[fmri], y)
                print(", Done in %.3f min" % ((time.time()-start)/60))

        X_connectome = self._transform(X_df)
        # Y, finalmente, coge y rellena los pesos del pca.
        self.pca.fit(X_connectome)

        return self

    def _transform(self, X_df):
        """Esta función simplemente coge y te crea las matrices de connectome.
        """
        # En mi opinión, definir otra vez el nombre de los archivos es un poco
        # tontaina, pero bueno...
        fmri_filenames = {col: X_df[col]
                          for col in X_df.columns if col in self.fmri_names}
        X_connectome = []
        for fmri in fmri_filenames:
            print("Transforming", fmri, end="")
            start = time.time()
            X_connectome.append(self.fmri_transformers[fmri].transform(fmri_filenames[fmri]))
            print(", Done in %.3f min" % ((time.time()-start)/60))
        return np.concatenate(X_connectome, axis=1)

    def transform(self, X_df):
        # Aquí se construye el dataframe de los pca de los connectomes
        X_connectome = self.pca.transform(self._transform(X_df))
        X_connectome = pd.DataFrame(X_connectome, index=X_df.index)
        # Le damos el nombre de connectome_i a cada columna
        X_connectome.columns = ['connectome_{}'.format(i)
                                for i in range(X_connectome.columns.size)]
        # get the anatomical information
        # Ojito. Indica la columna participants_age con doble corchete.
        # Eso hace que se quede X_part como un dataframe en lugar de una
        # serie de pandas.
        X_part = X_df[["participants_age"]]
        # OTRO TRUCAZO. Fíjate cómo utiliza la lambda en combinación con el map
        # La lambda no es otra cosa que hacer una definición de función en una
        # sola línea para mejor lectura y el mapeo es sacar un objeto del mismo
        # tipo y dimensión que el objeto origen
        X_part["participants_sex"] = X_df["participants_sex"].map(lambda x: 0 if x=="M" else 1)
        # Menudo parguela... Le cambia el nombre a las columnas del dataframe
        # X_part, pero los pone al revés. Debería ser
        # X_part.columns = ['anatomy_age', 'anatomy_sex']
        X_part.columns = ['anatomy_sex', 'anatomy_age']
        # Esto es lo mismo que el starting kit
        X_anatomy = X_df[[col for col in X_df.columns
                          if col.startswith('anatomy')]]
        X_anatomy = X_anatomy.drop(columns='anatomy_select')
        # concatenate both matrices
        # Total, lo único que ha hecho ha sido hacer pca a los conectomes de
        # casi todos los atlases y meter en el dataset la edad y el sexo
        # (como 0 o 1).
        return pd.concat([X_connectome, X_anatomy, X_part], axis=1)
