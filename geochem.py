import pandas as pd
import periodictable as pt
from compositions import renormalise

def simple_oxides(cation, output='formula'):
    """
    Creates a list of oxides for a cationic element
    (oxide of ions with c=1+ and above).
    """
    if not isinstance(cation, pt.core.Element):
        catstr = titlecase(cation)  # edge case of lowercase str such as 'cs'
        cation = getattr(pt, catstr)

    ions = [c for c in cation.ions if c > 0]  # Use only positive charges
    oxides = [pt.formula(f'{cation}{1}O{c//2}') if not c%2
              else pt.formula(f'{cation}{2}O{c}') for c in ions]
    if not output == 'formula':
        oxides = [ox.__str__() for ox in oxides]
    return oxides


def common_elements(cutoff=93, output='formula'):
    """
    Provides a list of elements up to a particular cutoff (default: including U)
    Output options are 'formula', or strings.
    """
    elements = [el for el in pt.elements
                if not (el.__str__() == 'n' or el.number>=cutoff)]
    if not output == 'formula':
        elements = [el.__str__() for el in elements]
    return elements


def REE_elements(output='formula'):
    """
    Provides the list of Rare Earth Elements
    Output options are 'formula', or strings.
    """
    elements = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
            'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
    if not output == 'formula':
        elements = [getattr(pt, el) for el in elements]
        elements = [el.__str__() for el in elements]
    return elements


def common_oxides(elements: list=[], output='formula',
                  addition: list=['FeOT', 'Fe2O3T', 'LOI']):
    """
    Creates a list of oxides based on a list of elements.
    Output options are 'formula', or strings.
    """
    if not elements:
        elements = [el for el in common_elements()
                    if not el.__str__() == 'O']  # Exclude oxygen
    oxides = [ox for el in elements
              for ox in simple_oxides(el)] + addition
    if not output == 'formula':
        oxides = [ox.__str__() for ox in oxides]
    return oxides


def devolatilise(df: pd.DataFrame,
                 exclude=['H2O', 'H2O_PLUS', 'H2O_MINUS', 'CO2', 'LOI'],
                 renorm=True):
    """
    Recalculates components after exclusion of volatile phases (e.g. H2O, CO2).
    """
    keep = [i for i in df.columns if not i in exclude]
    if renorm:
         return renormalise(df.loc[:, keep])
    else:
        return df.loc[:, keep]


def swap_simple_oxide(oxin, oxout):
    """
    Generates a function to convert oxide components between
    two elemental oxides, for use in redox recalculations.
    """
    inatoms = {k: v for (k, v) in oxin.atoms.items() if not k.__str__()=='O'}
    outatoms =  {k: v for (k, v) in oxout.atoms.items() if not k.__str__()=='O'}
    assert len(inatoms) == len(outatoms) == 1  # Assertion of simple oxide
    cation_coefficient = list(outatoms.values())[0] / list(inatoms.values())[0]
    def swap(dfser: pd.Series, molecular=False):
        if not molecular:
            swapped = dfser * cation_coefficient \
                        * oxout.mass / oxin.mass
        else:
            swapped = dfser * cation_coefficient
        return swapped

    return swap


def recalculate_redox(df: pd.DataFrame,
                      to_oxidised=False,
                      renorm=True,
                      total_suffix='T'):
    """
    Recalculates abundances of redox-sensitive components (particularly Fe),
    and normalises a dataframe to contain only one oxide species for a given
    element.
    """
    # Assuming either (a single column) or (FeO + Fe2O3) are reported
    # Fe columns - FeO, Fe2O3, FeOT, Fe2O3T
    FeO = pt.formula("FeO")
    Fe2O3 = pt.formula("Fe2O3")
    dfc = df.copy()
    ox_species = ['Fe2O3', f"Fe2O3{total_suffix}"]
    ox_in_df = [i for i in ox_species if i in dfc.columns]
    red_species = ['FeO', f"FeO{total_suffix}"]
    red_in_df = [i for i in red_species if i in dfc.columns]
    if to_oxidised:
        oxFe = swap_simple_oxide(FeO, Fe2O3)
        Fe2O3T = dfc.loc[:, ox_in_df].fillna(0).sum(axis=1) + \
                 oxFe(dfc.loc[:, red_in_df].fillna(0)).sum(axis=1)
        dfc.loc[:, 'Fe2O3T'] = Fe2O3T
        to_drop = red_in_df + \
                  [i for i in ox_in_df if not i.endswith(total_suffix)]
    else:
        reduceFe = swap_simple_oxide(Fe2O3, FeO)
        FeOT = dfc.loc[:, red_in_df].fillna(0).sum(axis=1) + \
               reduceFe(dfc.loc[:, ox_in_df].fillna(0)).sum(axis=1)
        dfc.loc[:, 'FeOT'] = FeOT
        to_drop = ox_in_df + \
                  [i for i in red_in_df if not i.endswith(total_suffix)]

    dfc = dfc.drop(columns=to_drop)

    if renorm:
        return renormalise(dfc)
    else:
        return dfc


def to_molecular(df: pd.DataFrame, renorm=True):
    """
    Converts mass quantities to molar quantities of the same order.
    E.g.:
    mass% --> mol%
    mass-ppm --> mol-ppm
    """
    MWs = [pt.formula(c).mass for c in df.columns]
    if renorm:
         return renormalise(df.div(MWs))
    else:
        return df.div(MWs)


def to_weight(df: pd.DataFrame, renorm=True):
    """
    Converts molar quantities to mass quantities of the same order.
    E.g.:
    mol% --> mass%
    mol-ppm --> mass-ppm
    """
    MWs = [pt.formula(c).mass for c in df.columns]
    if renorm:
        return renormalise(df.multiply(MWs))
    else:
        return df.multiply(MWs)