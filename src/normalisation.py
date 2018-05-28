import pandas as pd
import numpy as np
from pathlib import Path
from compositions import *

RELMASSS_UNITS = {
                  '%': 10**-2,
                  'wt%': 10**-2,
                  'ppm': 10**-6,
                  'ppb': 10**-9,
                  'ppt': 10**-12,
                  'ppq': 10**-15,
                  }


def scale_function(in_unit, target_unit='ppm'):
    if not pd.isna(in_unit):
        return RELMASSS_UNITS[in_unit.lower()] / \
               RELMASSS_UNITS[target_unit.lower()]
    else:
        return 1.


class RefComp(object):
    """
    Reference compositional model object, principally used for normalisation.
    """

    def __init__(self, filename, **kwargs):
        self.data = pd.read_csv(filename, **kwargs)
        self.data = self.data.set_index('var')
        self.original_data = self.data.copy() # preserve unaltered record
        self.add_oxides()
        self.collect_vars()
        self.set_units()

    def add_oxides(self):
        """
        Compositional models typically include elements in both oxide and elemental form,
        typically divided into 'majors' and 'traces'.

        For the purposes of normalisation - we need
            i) to be able to access values for the form found in the sample dataset,
            ii) for original values and uncertanties to be preserved, and
            iii) for closure to be preserved.

        There are multiple ways to acheive this - one is to create linked element-oxide tables,
        and another is to force working in one format (i.e. Al2O3 (wt%) --> Al (ppm))
        """
        pass

    def collect_vars(self,
                     headers=['Reservoir', 'Reference', 'ModelName', 'ModelType'],
                     floatvars=['value', 'unc_2sigma', 'constraint_value']):
        self.vars = [i for i in self.data.index if (not pd.isna(self.data.loc[i, 'value'])) and (i not in headers)]
        self.data.loc[self.vars, floatvars] = self.data.loc[self.vars, floatvars].apply(pd.to_numeric, errors='coerce')

    def set_units(self, to='ppm'):
        self.data.loc[self.vars, 'scale'] = \
            self.data.loc[self.vars, 'units'].apply(scale_function,
                                                    target_unit=to)
        self.data.loc[self.vars, 'value'] *= self.data.loc[self.vars, 'scale']
        self.data.loc[self.vars, 'units'] = to

    def normalize(self, df, aux_cols=["LOD","2SE"]):
        """
        Normalize the values within a dataframe to the refererence composition.
        Here we create indexes for normalisation of values and any auxilary values (e.g. uncertainty).
        """
        dfc = df.copy()
        cols = [v for v in self.vars if v in df.columns]
        mdl_ix = cols.copy()
        df_cols = cols.copy()
        for c in aux_cols:
            df_cols += [v+c for v in cols if v+c in dfc.columns]
            mdl_ix += [v for v in cols if v+c in dfc.columns]

        dfc.loc[:, df_cols] = np.divide(dfc.loc[:, df_cols].values,
                                        self.data.loc[mdl_ix, 'value'].values * \
                                        self.data.loc[mdl_ix, 'scale'].values)
        return dfc

    def __getattr__(self, var):
        """
        Allow access to model values via attribute e.g. Model.Si
        """
        return self.data.loc[var, 'value']

    def __getitem__(self, vars):
        """
        Allow access to model values via [] indexing e.g. Model['Si', 'Cr']
        """
        return self.data.loc[vars, ['value', 'unc_2sigma', 'units']]

    def __repr__(self):
        return f"Model of {self.Reservoir} ({self.Reference})"


def build_reference_db(directory, formats=['csv'], **kwargs):
    """
    Build all reference models in a given directory.
    """
    assert directory.exists() and directory.is_dir()

    files = []
    for fmt in formats:
        files.extend(directory.glob("./*."+fmt))

    comps = {}
    for f in files:
        r = RefComp(f, **kwargs)
        comps[r.ModelName] = r
    return comps
