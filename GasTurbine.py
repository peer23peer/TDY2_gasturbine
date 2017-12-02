import numpy as np
from math import nan
import matplotlib.pyplot as plt
from decimal import *
from pint import UnitRegistry, set_application_registry
import pint.quantity

u = UnitRegistry(autoconvert_offset_to_baseunit=True)
u.setup_matplotlib(True)
u.default_format = '~P'
set_application_registry(u)

np.set_printoptions(precision=3)

# States
V_flux = [0. for n in range(4)] * u.m ** 3 / u.s
q = {'0->1': nan * u.kJ / u.kg,
     '1->2': nan * u.kJ / u.kg,
     '2->3': nan * u.kJ / u.kg,
     '3->0': nan * u.kJ / u.kg}
w = {'0->1': nan * u.kJ / u.kg,
     '1->2': nan * u.kJ / u.kg,
     '2->3': nan * u.kJ / u.kg,
     '3->0': nan * u.kJ / u.kg}

P = [0. for n in range(4)] * u.Pa
T = [0. for n in range(4)] * u.K
rho = [0. for n in range(4)] * u.kg / u.m ** 3
m_flux = 0 * u.kg / u.s
v = [0. for n in range(4)] * u.m ** 3 / u.kg
c = [0. for n in range(4)] * u.m / u.s

# Given values
c_p = 1005. * u.J / (u.kg * u.K)
c_v = 716. * u.J / (u.kg * u.K)
P_atm = 1.013 * u.bar
P[0] = P_atm
T[0] = (15. * u.degC).to(u.K)
A = [1., 0.5, 0.1, 1.] * u.m ** 2

# Variables
V_flux[0] = 6000. * u.m ** 3 / u.hr
P[1] = 10. * u.bar
Q_in = 2000 * u.kW

# Calculated constants
R = c_p - c_v
k = c_p / c_v


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


def isobaar_dV(c, A):
    return c * A


# @u.wraps(ret='K', args=('K', 'm**3/kg', 'm**3/kg'))
def isobaar_dT(T_s, v_s, v_e):
    r"""
    \begin{equation}
        \frac{v_1}{T_1} = \frac{v_2}{T_2} \rightarrow T_2 = \frac{v_2}{v_1} T_1
    \end{equation}
    """
    return T_s * (v_e / v_s)


m_flux = mass_flux(P[0], V_flux[0], R, T[0])
print(repr(m_flux))
rho[0] = density(P[0], R, T[0])
v[0] = specific_volume(m_flux, V_flux[0])
c[0] = speed(m_flux, rho[0], A[0])

# 0 -> 1
T[1] = isentropische_dT(T[0], P[0], P[1], k)
rho[1] = density(P[1], R, T[1])
V_flux[1] = isentropische_dV(P[0], P[1], V_flux[0], k)
v[1] = isentropische_dv(V_flux[1], m_flux)
q['0->1'] = 0. * u.kJ / u.kg
w['0->1'] = isentropisch_work(P[0], v[0], P[1], v[1], k)
c[1] = speed(m_flux, rho[1], A[1])

# 1 -> 2
P[2] = P[1]
q['1->2'] = isobaar_heat(Q_in=Q_in, m_flux=m_flux)
w['1->2'] = isobaar_work(k=k, q=q['1->2'])
v[2] = isobaar_dv(q['1->2'], P[1], v[1], k)
T[2] = isobaar_dT(T[1], v[1], v[2])
rho[2] = density(P[2], R, T[2])
c[2] = speed(m_flux, rho[2], A[2])
V_flux[2] = isobaar_dV(c[2], A[2])

# 2 -> 3
P[3] = P[0]
T[3] = isentropische_dT(T[2], P[2], P[3], k)
rho[3] = density(P[3], R, T[3])
V_flux[3] = isentropische_dV(P[2], P[3], V_flux[2], k)
v[3] = isentropische_dv(V_flux[3], m_flux)
q['2->3'] = 0. * u.kJ / u.kg
w['2->3'] = isentropisch_work(P[2], v[2], P[3], v[3], k)
c[3] = speed(m_flux, rho[3], A[3])

# 3 -> 0
w['3->0'] = isobaar_work(R=R, T_s=T[3], T_e=T[0])
q['3->0'] = isobaar_heat(c_p=c_p, T_s=T[3], T_e=T[0])

# delta U
dU = {}
for key, value in q.items():
    dU[key] = value - w[key]

print(rho)
print(v ** -1)
print(T)
