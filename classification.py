import os
import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd
import pickle
from sklearn.externals import joblib

logger = logging.getLogger(__name__)

class ClassifierModel(object):

    def __init__(self, rebuild=False, deterministic=False):
        self.clsf_modelname = f"{type(self).__name__}"
        if __name__ == '__main__':
            modeldir = Path(os.getcwd()) / 'models'
        else:
            modeldir = Path(__file__).resolve().parent / 'models'
        self.diskname =  modeldir /  f'{self.clsf_modelname}'
        self.deterministic = deterministic
        self.clsf = None
        # Try load classifer
        if rebuild: self.rebuild_clsf()
        try:
            self.clsf = self.load_clsf(self.diskname.with_suffix('.clsf.gz'))
            logger.info(f'Loaded {self.diskname}')
        except:
            self.rebuild_clsf()

    def rebuild_clsf(self):
        self.clsf = self.build_clsf()
        self.dump_clsf(self.diskname.with_suffix('.clsf.gz'))
        logger.info(f'Built {self.clsf_modelname}')

    def load_clsf(self, file):
        assert file.exists() and file.is_file()
        logger.info(f'Loading {self.diskname}')
        return joblib.load(file)

    def build_clsf(self):
        logger.info(f'Building {self.clsf_modelname}')
        clsf = None
        return clsf

    def dump_clsf(self, filename):
        joblib.dump(self.clsf, filename, compress=('gzip', 3))
        logger.info(f'Dumped {filename}')
        assert filename.exists() and filename.is_file()

    def classify(self, df:pd.DataFrame):
        return self.clsf.predict(df)

    def __str__(self):
        pass

    def __repr__(self):
        pass


class SimpleDeterministicClassifer(object):

    def __init__(self, parent:ClassifierModel):
        self.parent=parent
        self.fields=None
        self.load_fields()

    def load_fields(self):
        if self.parent:
            self.clsf_modelname = self.parent.clsf_modelname
            self.modeldir = self.parent.diskname.resolve()
            af = self.modeldir / f'{self.clsf_modelname}.allfields'
            if af.exists() and af.is_file():
                logger.info(f'''Loading {self.clsf_modelname} classifer fields from "allfile".''')
                loadup = joblib.load(af)
                self.fclasses =  [k for (k, v) in loadup.items()]
                self.fields = {c: loadup[c] for ix, c in enumerate(self.fclasses)}
            else:
                logger.info(f'Loading {self.clsf_modelname} classifer fields from files.')
                fieldfiles = self.modeldir.glob(f'{self.clsf_modelname}.*')
                loadup = [joblib.load(f) for f in fieldfiles]
                self.fclasses =  [f.suffix.replace('.', "") for f in loadup]
                self.fields = {c: loadup[ix] for ix, c in enumerate(self.fclasses)}
            logger.info(f'''Loaded {self.clsf_modelname} classifer fields.''')
        else:
            pass

    def add_to_axes(self, ax=None, **kwargs):
        polys = [(c, self.fields[c]) for c in self.fclasses if self.fields[c]['poly']]
        if ax is None:
            fig, ax = plt.subplots(1)
        pgns = []
        for c, f in polys:
            label = f['names']
            pg = Polygon(f['poly'], closed=True, **kwargs)
            pgns.append(pg)
            x, y = pg.get_xy().T
            ax.annotate(c, xy=(np.nanmean(x), np.nanmean(y)))
            ax.add_patch(pg)
        #ax.add_collection(PatchCollection(pgns), autolim=True)

    def predict(self, df: pd.DataFrame, cols:list=['x', 'y']):
        points = df.loc[:,cols].values
        polys = [Polygon(self.fields[c]['poly'], closed=True)
                 for c in self.fclasses] # if self.fields[c]['poly']

        indexes = np.array([p.contains_points(points) for p in polys]).T
        notfound = np.logical_not(indexes.sum(axis=1))
        out = pd.Series(index=df.index)
        outlist = list(map(lambda ix: self.fclasses[ix], np.argmax(indexes, axis=1)))
        out.loc[:] = outlist
        out.loc[(notfound)] = 'none'
        return out

class PeralkalinityClassifier(object):
    def __init__(self, parent:ClassifierModel):
        self.parent=parent
        self.fields=None

    def predict(self, df: pd.DataFrame):
        TotalAlkali = df.Na2O + df.K2O
        perkalkaline_where = (df.Al2O3 < (TotalAlkali + df.CaO) and
                              TotalAlkali > df.Al2O3)
        metaluminous_where = (df.Al2O3 > (TotalAlkali + df.CaO) and
                              TotalAlkali < df.Al2O3)
        peraluminous_where = (df.Al2O3 < (TotalAlkali + df.CaO) and
                              TotalAlkali < df.Al2O3)
        out = pd.Series(index=df.index)
        out.loc[peraluminous_where] = 'Peraluminous'
        df.loc[metaluminous_where] = 'Metaluminous'
        df.loc[perkalkaline_where] = 'Peralkaline'
        return out



class Geochemistry(object):


    class peralkalinity(ClassifierModel):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, deterministic=True, **kwargs)

        def build_clsf(self):

            return PeralkalinityClassifier(self)

    class TAS(ClassifierModel):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, deterministic=True, **kwargs)

        def build_clsf(self):
            # fields are functions which define ranges where they are invalid
            # Encode Field Names
            #   LabelBinarizer
            # Create Field
            # Ensure non-valid entries are classified as nan/invalid
            # Build
            return SimpleDeterministicClassifer(self)

        def add_to_axes(self, ax=None, **kwargs):
            if ax is None:
                fig, ax = plt.subplots(1)
            self.clsf.add_to_axes(ax=ax, **kwargs)
            ax.set_xlim((35, 85))
            ax.set_ylim((0, 20))

        def classify(self, df:pd.DataFrame):
            return self.clsf.predict(df, cols=['SiO2', 'TotalAlkali'])


class Petrology(object):


    class apahnitic(ClassifierModel):
        """
        QAPF Diagram
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, deterministic=True, **kwargs)

        def build_clsf(self):
            # Build
            clsf = 'thing'
            return clsf

    class phaneritic(ClassifierModel):
        """
        QAPF Diagram
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, deterministic=True, **kwargs)

        def build_clsf(self):
            # Build
            clsf = 'thing'
            return clsf

    class gabbroic(ClassifierModel):
        """
        Pyroxene-Olivine-Plagioclase
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, deterministic=True, **kwargs)

        def build_clsf(self):
            # Build
            clsf = 'thing'
            return clsf

    class ultramafic(ClassifierModel):
        """
        Olivine-Orthopyroxene-Clinopyroxene
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, deterministic=True, **kwargs)

        def build_clsf(self):
            # Build
            clsf = 'thing'
            return clsf


if __name__ == '__main__':
    #%matplotlib inline
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
    cm = Geochemistry.TAS(rebuild=False)
    #cm = Petrology.apahnitic()
    fig, ax = plt.subplots(1)
    cm.add_to_axes(ax=ax, alpha=0.4, color='k')
    plt.show()

    from compositions import renormalise
    c2 = Geochemistry.peralkalinity()
    df = pd.DataFrame(index=np.arange(100))
    df['Al2O3'] = np.linspace(20, 50, len(df.index))
    df['Na2O'] = np.linspace(20, 10, len(df.index))
    df['K2O'] = np.linspace(10, 122, len(df.index))
    df['CaO'] = np.linspace(5, 30, len(df.index))
    df = renormalise(df)
    c2.classify(df)
