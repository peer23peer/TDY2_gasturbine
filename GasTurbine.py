import numpy as np
from IPython.display import Latex
from enum import Enum
from collections import UserList
from pint import UnitRegistry, set_application_registry

u = UnitRegistry(autoconvert_offset_to_baseunit=True)
u.setup_matplotlib(True)
u.default_format = '.3~L'
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

    def __str__(self):
        return r'State: {} \\' \
               r'P: {:.3~L} \\' \
               r'T: {:.3~L} \\' \
               r'v: {:.3~L} \\' \
               r'\dot{{V}}: {:.3~L}\\' \
               r'\rho: {:.3~L} \\' \
               r'c: {:.3~L}'.format(self.id, self.P.to('Pa'), self.T.to('K'), self.v.to('m**3/kg'),
                                    self.V_flux.to('m**3/s'), self.rho.to('kg/m**3'), self.c.to('m/s'))


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

    def __str__(self):
        t_f = '{}-{}'.format(self.start.id, self.end.id)
        return r'Transition: {t_f} \\' \
               r'q_{{{t_f}}}: {q:.3~L} \\' \
               r'w_{{{t_f}}}: {w:.3~L} \\' \
               r'w_{{t,{t_f}}}: {wt:.3~L} \\' \
               r'\Delta u_{{{t_f}}}: {du:.3~L}\\' \
               r'\Delta e_{{{t_f},kin}}: {dekin:.3~L} \\' \
               r'\Delta e_{{{t_f},pot}}: {depot:.3~L} \\' \
               r'\Delta h_{{{t_f}}}: {dh:.3~L}'.format(t_f=t_f, q=self.q, w=self.w,
                                                       wt=self.w_t, du=self.du, dekin=self.de_kin,
                                                       depot=self.de_pot, dh=self.dh)


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
    def transitions(self):
        return self._transitions

    @transitions.setter
    def transitions(self, value):
        self._transitions = [Transition() for t in range(len(value))]
        for t, s, e in self:
            t.start = value[s]
            t.end = value[e]

    def __iter__(self):
        from_id = [i for i in range(len(self._transitions))]
        to_id = [from_id[1:], [from_id[0]]]
        to_id = [y for x in to_id for y in x]
        for t, s, e in zip(self.transitions, from_id, to_id):
            yield t, s, e

    def sum_q(self):
        return sum([t.q for t in self.transitions])


def latex(value):
    return Latex(r'$ {} $'.format(value))


# Standard functions:
def density(P, R, T):
    r"""
    \begin{equation}
        \rho_0 = \frac{P_0}{T_0 R}
        \begin{cases}
            \frac{P_0 v_0}{T_0} = \dot{m} R \\
            \rho_0 = \frac{\dot{m}}{v_0}
        \end{cases}
    \end{equation}
    """
    return P / (R * T)


# @u.wraps(ret='kg/s', args=('Pa', 'm**3/s', 'J/(kg*K)', 'K'))
def mass_flux(P, V_flux, R, T):
    r"""
    \begin{equation}
        \frac{P_0 \dot{V_0}}{T_0} = \dot{m} R \rightarrow \dot{m} = \frac{P_0 \dot{V_0}}{R T_0}
    \end{equation}
    """
    return (P * V_flux / (R * T)).to('kg/s')


# @u.wraps(ret='m**3/kg', args=('kg/s', 'm**3/s'))
def specific_volume(m_flux, V_flux):
    r"""
    \begin{equation}
        v_0 = \frac{\dot{V_0}}{\dot{m}}
    \end{equation}
    """
    return V_flux / m_flux


# @u.wraps(ret='m/s', args=('kg/s', 'kg/m**3', 'm**2'))
def speed(m_flux, rho, A):
    r"""
    \begin{equation}
        c_0 A_0 \rho_0 = \dot{m} \rightarrow c_0 = \frac{\dot{m}}{\rho_0 A_0}
    \end{equation}
    """
    return m_flux / (rho * A)


# @u.wraps(ret='m/s', args=('m/s', 'm**2', 'kg/m**3', 'm**2', 'kg/m**3'))
def Delta_speed(c_s, A_s, rho_s, A_e, rho_e):
    r"""
    \begin{equation}
        c_0 A_0 \rho_0 = c_1 A_1 \rho_1 \rightarrow c_1 = \frac{A_0 \rho_0}{A_1 \rho_1} c_0
    \end{equation}
    """
    return (A_s * rho_s) / (A_e * rho_e) * c_s


# @u.wraps(ret='kJ/kg', args=('Pa', 'm**3/kg', 'Pa', 'm**3/kg', ''))
def isentropisch_work(P_s, V_s, P_e, V_e, k):
    r"""
    \begin{equation}
        w_{0-1} = \frac{k}{k-1}\left(P_1 v_1 - P_0 v_0\right)
    \end{equation}
    """
    return (-1. / (k - 1) * (P_e * V_e - P_s * V_s)).to('kJ/kg')


# @u.wraps(ret='K', args=('K', 'Pa', 'Pa', None))
def isentropische_dT(T_s, P_s, P_e, k):
    r"""
    \begin{equation}
        \frac{T_0^k}{P_0^{k-1}} = \frac{T_1^k}{P_1^{k-1}} \rightarrow T_1 = \left(\left(\frac{P_1}{P_0}\right)^{k-1}T_0^k\right)^{\frac{1}{k}}
    \end{equation}
    """
    return ((P_e / P_s) ** (k - 1) * T_s ** k) ** (1 / k)


@u.wraps(ret='m**3/s', args=('Pa', 'Pa', 'm**3/s', None))
def isentropische_dV(P_s, P_e, V_flux_s, k):
    r"""
    \begin{equation}
        p_0 V_0^k = p_1 V_1^k \rightarrow v_1 = \left(\frac{P_0}{P_1}v_0^k\right)^{\frac{1}{k}}
    \end{equation}
    """
    return ((P_s / P_e) * V_flux_s ** k) ** (1 / k)


# @u.wraps(ret='m**3/kg', args=('m**3/s', 'kg/s'))
def isentropische_dv(V_flux, m_flux):
    r"""
    \begin{equation}
        v_1 = \frac{\dot{V}_1}{\dot{m}}
    \end{equation}
    """
    return V_flux / m_flux


def isentropische_dP(P_s, v_s, v_e, k):
    return (P_s * (v_s / v_e) ** k).to('Pa')


# @u.wraps(ret='kJ/kg', args=('kW', 'kg/s'))
def _isobaar_heat_Qm(Q_in, m_flux):
    r"""
    \begin{equation}
        q_{1-2} = \frac{Q_{in}}{\dot{m}}
    \end{equation}
    """
    return (Q_in / m_flux).to('kJ/kg')


def _isobaar_heat_cpdT(c_p, T_s, T_e):
    return (c_p * (T_e - T_s)).to('kJ/kg')


def isobaar_heat(**kwargs):
    if 'Q_in' in kwargs.keys() and 'm_flux' in kwargs.keys():
        return _isobaar_heat_Qm(**kwargs)
    else:
        return _isobaar_heat_cpdT(**kwargs)


# @u.wraps(ret='kJ/kg', args=(None, 'kJ/kg'))
def _isobaar_work_kq(k, q):
    r"""
    \begin{equation}
        w_{1-2} = \frac{q_{1-2}}{\frac{k}{k-1}}
        \begin{cases}
            q_{1-2} = \frac{k}{k-1} P(v_2 - v_1) \\
            w_{1-2} = P(v_2 - v_1)
        \end{cases}
    \end{equation}
    """
    return (q / (k / (k - 1))).to('kJ/kg')


def _isobaar_work_R_dt(R, T_s, T_e):
    return (R * (T_e - T_s)).to('kJ/kg')


def isobaar_work(**kwargs):
    if 'k' in kwargs.keys() and 'q' in kwargs.keys():
        return _isobaar_work_kq(**kwargs)
    else:
        return _isobaar_work_R_dt(**kwargs)


# @u.wraps(ret='m**3/kg', args=('kJ/kg', 'Pa', 'm**3/kg', ''))
def isobaar_dv(q, P, v_s, k):
    r"""
    \begin{equation}
        q_{1-2} = \frac{k}{k-1}P(v_2-v_1) \rightarrow v_2 = \frac{q_{1-2}}{\frac{k}{k-1}P} + v_1
    \end{equation}
    """
    return q / (P * k / (k - 1)) + v_s


def V_flux(rho_s, rho_e, V_s_flux):
    return V_s_flux * (rho_s / rho_e)


# @u.wraps(ret='K', args=('K', 'm**3/kg', 'm**3/kg'))
# def isobaar_dT(T_s, v_s, v_e):
#     r"""
#     \begin{equation}
#         \frac{v_1}{T_1} = \frac{v_2}{T_2} \rightarrow T_2 = \frac{v_2}{v_1} T_1
#     \end{equation}
#     """
#     return T_s * (v_e / v_s)
def isobaar_dT(q, c_p, T_s):
    return (q / c_p + T_s).to('K')


def technical_work(P_s, v_s, P_e, v_e, n):
    return ((-n / (n - 1)) * (P_e * v_e - P_s * v_s)).to('kJ/kg')


def delta_e_kin(c_s, c_e):
    return (0.5 * (c_e ** 2 - c_s ** 2)).to('kJ/kg')


def delta_h(q, w_t, de_kin):
    return (q - w_t - de_kin).to('kJ/kg')


def delta_u(q, w):
    return (q - w).to('kJ/kg')
