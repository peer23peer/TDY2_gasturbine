import numpy as np
import six
from IPython.display import Latex
import matplotlib.pyplot as plt
import pytablewriter
from enum import Enum
from collections import UserList
from pint import UnitRegistry, set_application_registry
from pint import __version__ as pint_version

u = UnitRegistry(autoconvert_offset_to_baseunit=True)
if int(pint_version.split('.')[0]) * 10 + int(pint_version.split('.')[1]) > 8:
    u.setup_matplotlib(True)

u.default_format = '.4f~L'
set_application_registry(u)

np.set_printoptions(precision=3)

__all__ = ['density', 'specific_volume', 'u', 'np', 'latex', 'V_flux', 'delta_u',
           'speed', 'Delta_speed', 'isentropisch_work', 'isentropische_dT', 'isentropische_dV',
           'isentropische_dv', 'isentropische_dP', 'isobaar_heat', 'isobaar_work',
           'isobaar_dv', 'isobaar_dT', 'technical_work', 'delta_e_kin',
           'delta_h', 'ProcessType', 'TransitionState', 'Transition', 'Cycle', 'States', 'mass_flux']


class ProcessType(Enum):
    POLYTROPIC = -1
    ISOBARIC = 0
    ISENTROPIC = 1


class TransitionState:
    P = 0. * u.Pa
    v = 0. * u.m ** 3 / u.kg
    rho = 0. * u.kg / u.m ** 3
    T = 0. * u.K
    c = 0. * u.m / u.s
    A = 0. * u.m ** 2
    V_flux = 0. * u.m ** 3 / u.s
    id = 0

    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return repr(self.id, self.P, self.T, self.v)

    def __str__(self):
        self.to_pref_units()
        return r'state: {} P:{:.3}, T:{:.3}, v:{:.3}'.format(self.id, self.P, self.T, self.v)

    def print(self):
        self.to_pref_units()
        t_f = '{}'.format(self.id)
        writer = pytablewriter.LatexMatrixWriter()
        writer.table_name = 'State_{{{t_f}}}'.format(t_f=t_f)
        writer.header_list = ['', 'value                ']
        writer.value_matrix = [
            ['P_{{{t_f}}}'.format(t_f=t_f), '{}'.format(self.P)],
            ['T_{{{t_f}}}'.format(t_f=t_f), '{}'.format(self.T)],
            ['v_{{{t_f}}}'.format(t_f=t_f), '{}'.format(self.v)],
            ['\\dot{{V}}_{{{t_f}}}'.format(t_f=t_f), '{}'.format(self.V_flux)],
            ['\\rho_{{{t_f}}}'.format(t_f=t_f), '{}'.format(self.rho)],
            ['c_{{{t_f}}}'.format(t_f=t_f), '{}'.format(self.c)]
        ]
        writer.stream = six.StringIO()
        writer.write_table()
        return Latex(writer.stream.getvalue())

    def to_pref_units(self):
        self.P.ito('Pa')
        self.T.ito('K')
        self.v.ito('m**3/kg')
        self.V_flux.ito('m**3/s')
        self.rho.ito('kg/m**3')
        self.c.ito('m/s')


class States(UserList):
    def __init__(self, nstates):
        self.data = [TransitionState(s) for s in range(4)]


class Transition:
    start: TransitionState = TransitionState(0)
    end: TransitionState = TransitionState(1)
    processtype: ProcessType = ProcessType.POLYTROPIC
    q = 0. * u.kJ / u.kg
    w = 0. * u.kJ / u.kg
    w_t = 0. * u.kJ / u.kg
    du = 0. * u.kJ / u.kg
    de_kin = 0. * u.kJ / u.kg
    de_pot = 0. * u.kJ / u.kg
    dh = 0. * u.kJ / u.kg
    resolution = 100
    k = 0

    def __repr__(self):
        return repr((self.start, ' -> ', self.end))

    def __str__(self):
        return '{}->{}'.format(self.start.id, self.end.id)

    def to_pref_units(self):
        self.q.ito('kJ/kg')
        self.w.ito('kJ/kg')
        self.w_t.ito('kJ/kg')
        self.du.ito('kJ/kg')
        self.de_kin.ito('kJ/kg')
        self.de_pot.ito('kJ/kg')
        self.dh.ito('kJ/kg')

    def print(self):
        self.to_pref_units()
        t_f = str(self)
        writer = pytablewriter.LatexMatrixWriter()
        writer.table_name = 'Trans_{{{}}}'.format(t_f)
        writer.header_list = ['', 'value']
        writer.value_matrix = [
            ['q_{{{t_f}}}'.format(t_f=t_f), '{} '.format(self.q)],
            ['w_{{{t_f}}}'.format(t_f=t_f), '{}'.format(self.w)],
            ['w_{{t,{t_f}}}'.format(t_f=t_f), '{}'.format(self.w_t)],
            ['\\Delta u_{{{t_f}}}'.format(t_f=t_f), '{}'.format(self.du)],
            ['\\Delta e_{{{t_f},kin}}'.format(t_f=t_f), '{}'.format(self.de_kin)],
            ['\\Delta e_{{{t_f},pot}}'.format(t_f=t_f), '{}'.format(self.de_pot)],
            ['\\Delta h_{{{t_f}}}'.format(t_f=t_f), '{}'.format(self.dh)]
        ]
        writer.stream = six.StringIO()
        writer.write_table()
        return Latex(writer.stream.getvalue())

    def P_isobaar(self, v):
        return np.ones((self.resolution,)) * self.start.P.to('Pa')

    @u.wraps(ret='Pa', args=(None, 'm**3/kg', 'Pa', 'm**3/kg', ''))
    def P_isentroop(self, v, P_s, v_s, k):
        return P_s * (v_s / v) ** k

    @property
    def P(self):
        if self.processtype == ProcessType.ISENTROPIC:
            return self.P_isentroop(self.v, self.start.P, self.start.v, self.k)
        elif self.processtype == ProcessType.ISOBARIC:
            return self.P_isobaar(self.v).to('Pa')
        return 0

    @property
    def v(self):
        return np.linspace(self.start.v.to('m**3/kg').m, self.end.v.to('m**3/kg').m, self.resolution) * u.m ** 3 / u.kg


class Cycle:
    _transitions = []
    c_p = 0. * u.J / (u.kg * u.K)
    c_v = 0. * u.J / (u.kg * u.K)
    Q_in = 0. * u.kW
    m_flux = 0. * u.kg / u.s

    @property
    def R(self):
        return self.c_p - self.c_v

    @property
    def k(self):
        return self.c_p / self.c_v

    @property
    def t(self):
        return self._transitions

    @t.setter
    def t(self, value):
        self._transitions = [Transition() for t in range(len(value))]
        for t, s, e in self:
            t.start = value[s]
            t.end = value[e]
            t.k = self.k

    def __iter__(self):
        from_id = [i for i in range(len(self._transitions))]
        to_id = [from_id[1:], [from_id[0]]]
        to_id = [y for x in to_id for y in x]
        for t, s, e in zip(self.t, from_id, to_id):
            yield t, s, e

    def plot_Pv(self):
        for t, s, e in self:
            plt.plot(t.v, t.P, label='{}-{}'.format(s, e))
        plt.legend()
        plt.grid(True)
        plt.title('Pressure vs specific volume')
        plt.tight_layout()
        plt.show()

    def print_closed(self):
        u.default_format = '.3f~P'
        writer = pytablewriter.LatexTableWriter()
        writer.table_name = 'E_{closed}'
        writer.header_list = ['', 'q ', 'Delta u', 'w']
        value_matrix = []
        for t, s, e in self:
            c = []
            c.append('{}-{}'.format(s, e))
            c.append('{}'.format(t.q))
            c.append('{}'.format(t.du))
            c.append('{}'.format(t.w))
            value_matrix.append(c)
        sum_c = ['Sigma', '{}'.format(self.sum('q')), '{}'.format(self.sum('du')),
                 '{}'.format(self.sum('w'))]
        value_matrix.append(sum_c)
        writer.value_matrix = value_matrix
        writer.stream = six.StringIO()
        writer.write_table()
        return Latex(writer.stream.getvalue())

    def print_open(self):
        u.default_format = '.3f~P'
        writer = pytablewriter.LatexTableWriter()
        writer.table_name = 'E_{open}'
        writer.header_list = ['', 'q', 'Delta h', 'w_t', 'Delta e_{kin}',
                              'Delta e_{pot}']
        value_matrix = []
        for t, s, e in self:
            c = []
            c.append('{}-{}'.format(s, e))
            c.append('{}'.format(t.q))
            c.append('{}'.format(t.dh))
            c.append('{}'.format(t.w_t))
            c.append('{}'.format(t.de_kin))
            c.append('{}'.format(t.de_pot))
            value_matrix.append(c)
        sum_c = ['Sigma', '{}'.format(self.sum('q')), '{}'.format(self.sum('dh')),
                 '{}'.format(self.sum('w_t')),
                 '{}'.format(self.sum('de_kin')), '{}'.format(self.sum('de_pot'))]
        value_matrix.append(sum_c)
        writer.value_matrix = value_matrix
        writer.stream = six.StringIO()
        writer.write_table()
        return Latex(writer.stream.getvalue())


    def sum(self, attr):
        return sum([getattr(t, attr) for t in self.t])


def latex(value):
    return Latex(r'$ {} $'.format(value))


# Standard functions:
def density(P, R, T):
    return P / (R * T)


def mass_flux(P, V_flux, R, T):
    return (P * V_flux / (R * T)).to('kg/s')


def specific_volume(m_flux, V_flux):
    return V_flux / m_flux


def speed(m_flux, rho, A):
    return m_flux / (rho * A)


def Delta_speed(c_s, A_s, rho_s, A_e, rho_e):
    return (A_s * rho_s) / (A_e * rho_e) * c_s


def isentropisch_work(P_s, V_s, P_e, V_e, k):
    return (-1. / (k - 1) * (P_e * V_e - P_s * V_s)).to('kJ/kg')


@u.wraps(ret='K', args=('K', 'Pa', 'Pa', ''))
def isentropische_dT(T_s, P_s, P_e, k):
    return ((P_e / P_s) ** (k - 1) * T_s ** k) ** (1 / k)


@u.wraps(ret='m**3/s', args=('Pa', 'Pa', 'm**3/s', None))
def isentropische_dV(P_s, P_e, V_flux_s, k):
    return ((P_s / P_e) * V_flux_s ** k) ** (1 / k)


def isentropische_dv(V_flux, m_flux):
    return V_flux / m_flux


def isentropische_dP(P_s, v_s, v_e, k):
    return (P_s * (v_s / v_e) ** k).to('Pa')


def _isobaar_heat_Qm(Q_in, m_flux):
    return (Q_in / m_flux).to('kJ/kg')


def _isobaar_heat_cpdT(c_p, T_s, T_e):
    return (c_p * (T_e - T_s)).to('kJ/kg')


def isobaar_heat(**kwargs):
    if 'Q_in' in kwargs.keys() and 'm_flux' in kwargs.keys():
        return _isobaar_heat_Qm(**kwargs)
    else:
        return _isobaar_heat_cpdT(**kwargs)


def _isobaar_work_kq(k, q):
    return (q - (q / k)).to('kJ/kg')


def _isobaar_work_R_dt(R, T_s, T_e):
    return (R * (T_e - T_s)).to('kJ/kg')


def isobaar_work(**kwargs):
    if 'k' in kwargs.keys() and 'q' in kwargs.keys():
        return _isobaar_work_kq(**kwargs)
    else:
        return _isobaar_work_R_dt(**kwargs)


def isobaar_dv(q, P, v_s, k):
    return q / (P * k / (k - 1)) + v_s


def V_flux(c, A):
    return c * A


def isobaar_dT(q, c_p, T_s):
    return (q / c_p + T_s).to('K')


def technical_work(q, dh, de_kin, de_pot):
    return (q - dh - de_kin - de_pot).to('kJ/kg')


def delta_e_kin(c_s, c_e):
    return (0.5 * (c_e ** 2 - c_s ** 2)).to('kJ/kg')


def delta_h(T_s, T_e, c_p):
    return (c_p * (T_e - T_s)).to('kJ/kg')


def delta_u(q, w):
    return (q - w).to('kJ/kg')
