from typing import Iterable, Sequence, Any, Union
import numpy as np
import re

from observation_model import Galaxy
from util import Vector


def _decode_vizier_line(line: str) -> tuple[str, dict[str, Any]]:
    data_model = (
        ('Name', 0, 17, str),
        ('Survey', 18, 26, str),
        ('Class', 27, 29, int),
        ('RAdeg', 29, 37, float),
        ('DEdeg', 38, 46, float),
        ('m-M', 47, 51, float),
        ('ah', 52, 58, float),
        ('l_ell', 59, 61, str),
        ('ell', 60, 65, float),
        ('Dist', 66, 69, float),
        ('r1/2', 70, 74, float),
        ('VMag', 75, 81, float),
        ('Ref', 82, 87, str)
    )
    read = dict()
    for name, b, e, t in data_model:
        try:
            data = t(line[b:e].strip(' \n'))
            read[name] = data
        except ValueError:
            read[name] = None

    read['Name'] = read['Name'].strip(' ')
    return read.pop('Name'), read


class TexDecoder:
    def __init__(self, names: Sequence[str]):
        self.name_patt = r'[A-z][a-zA-z1-9]+'
        self.value_patt = r'\t([0-9+-.]+|\$-?{[\d.+-]+}_{[\d.+-]+}\^{[\d.+-]+}\$|cdots)'
        self.line_patt = r'^(%s)(%s){%i}\t?$' % (self.name_patt, self.value_patt, len(names))
        self.names = names

    def validate_line(self, line: str) -> bool:
        return re.match(self.line_patt, line) is not None

    def process_line(self, line: str) -> Union[tuple[str, dict[str, float]], None]:
        m1 = re.match(self.name_patt, line)
        if m1 is None:
            return None
        name = m1.group(0).strip(' ')
        gal = {}
        m2 = re.findall(self.value_patt, line)
        for i, m in enumerate(m2):
            match = re.match(r'[0-9+-.]+', m)
            if match is not None:
                gal[self.names[i]] = float(match.group(0))
                continue
            match = re.match(r'\$(-?{[\d.+-]+})_{([\d.+-]+)}\^{([\d.+-]+)}', m)
            if match is not None:
                val = float(re.sub(
                    r'[{}]',
                    '',
                    match.group(1)
                ))
                err_low = float(match.group(2))
                err_high = float(match.group(3))
                gal[self.names[i]] = val
                continue
            gal[self.names[i]] = None

        return name, gal


def _ar_to_rom(s: str) -> str:
    ar_rom = {
        1: 'I',
        2: 'II',
        3: 'III',
        4: 'IV',
        5: 'V',
        6: 'VI',
        7: 'VII',
        8: 'VIII',
        9: 'IX'
    }
    pat = r'([a-zA-Z])([\d])$'
    match = re.search(pat, s)
    if match is None:
        return s
    num = int(match.group(2))
    return re.sub(pat, f"\\1 {ar_rom[num]}", s)


def load_vizier(files: Iterable[str], vel: Iterable[str]) -> dict[str, Galaxy]:
    gals = dict()
    for file in files:
        with open(file, 'r') as f:
            for line in f.readlines():
                name, gal = _decode_vizier_line(line)
                gals[name] = gal

    dec = TexDecoder([
        'RADeg',
        'DEDeg',
        'm-M',
        'arcmin',
        'e',
        'theta',
        'Vh',
        'sigma',
        'FeH'
    ])
    sec_tab = dict()
    for file in vel:
        with open(file, 'r') as f:
            for line in f.readlines():
                if dec.validate_line(line):
                    name, res = dec.process_line(line)
                    # arabic numbers to Roman
                    name = _ar_to_rom(name)
                    # camel case to whitespaces
                    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
                    if gals.get(name) is not None:
                        gals[name]['Vel'] = res['Vh']
                    sec_tab[name] = res
    galaxies = {name: Galaxy(
        Vector.get_sph([
            gal['Dist'],
            gal['DEdeg'] / 180. * np.pi,
            gal['RAdeg'] / 180. * np.pi,
        ]),
        gal['Vel']
    ) for name, gal in gals.items() if
        gal.get('Dist') is not None and
        gal.get('DEdeg') is not None and
        gal.get('RAdeg') is not None and
        gal.get('Vel') is not None
    }
    keys = set(gals.keys())
    keys.update(sec_tab.keys())
    """print('Name                 Present      m-M difference Velocity')
    for key in keys:
        in_viz = False
        if gals.get(key) is not None:
            in_viz = True
            present = 'Vizier'
        in_sec = False
        if sec_tab.get(key) is not None:
            in_sec = True
            present = 'Other'
        m_m_diff = None
        if in_sec and in_viz:
            m_m_diff = gals[key]['m-M'] - sec_tab[key]['m-M']
            m_m_diff = round(m_m_diff, 3)
            present = 'Both'
        vel = False
        if in_sec:
            vel = sec_tab[key].get('Vh') is not None
        print('%20s|%12s|%14s|%8s' % (
            str(key),
            str(present),
            m_m_diff,
            str(vel)
        ))"""
    return galaxies
