import re

def quoted_string(s):
    #if " " in s or '-' in s or '_' in s:
    s = f'''"{s}"'''
    return s


def titlecase(s,
              exceptions=['and', 'in', 'a'],
              abbrv=['ID', 'IGSN', 'CIA', 'CIW',
                     'PIA', 'SAR', 'SiTiIndex', 'WIP'],
              capitalize_first=True,
              split_on="\s+",
              delim=""):
    """
    Formats strings in CamelCase, with exceptions for simple articles
    and omitted abbreviations which retain their capitalization.
    TODO: Option for retaining original CamelCase.
    """
    words = re.split(split_on, s)
    out=[]
    first = words[0]
    if capitalize_first and not (first in abbrv):
        first = first.capitalize()

    out.append(first)
    for word in words[1:]:
        out.append(word if word in exceptions+abbrv else word.capitalize())
    return delim.join(out)
