import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", False)

EPSILON = jnp.array(1.0e-15)
TFREEZ = jnp.array(273.15)
T0 = jnp.array(273.15)

LVH2O = jnp.array(2.501e6)
LSUBS = jnp.array(2.83e6)
LSUBF = jnp.array(3.335E+5)
XLF = jnp.array(3.50E5)
XLV = jnp.array(2.5E6)
XLS = jnp.array(2.85e6)

CPH2O = jnp.array(4.218E+3)
CPICE = jnp.array(2.106E+3)
CAIR = jnp.array(1004.0)
CLIQ = jnp.array(4190.0)
CICE = jnp.array(2106.0)

R = jnp.array(287.04)
RD = jnp.array(287.04)
R_D = jnp.array(287.0)
R_V = jnp.array(461.6)

CP = jnp.array(7 * 287 / 2)
CV = CP - R_D
CPV = 4.0 * R_V
CVV = CPV - R_V

CVPM = -CV / CP
RCV = R_D / CV
RCP = R_D / CP
CPOVCV = CP / (CP - R_D)
CVOVCP = 1.0 / CPOVCV
RVOVRD = R_V / R_D
EP_1 = R_V / R_D - 1.0
EP_2 = R_D / R_V
P608 = RVOVRD - 1.0

RHOWATER = jnp.array(1000.)
RHOSNOW = jnp.array(100.0)
RHOAIR0 = jnp.array(1.28)

SIGMA = jnp.array(5.67E-8)
STBOLT = jnp.array(5.67051E-8)
EMISSI_S = jnp.array(0.95)
KARMAN = jnp.array(0.4)

P1000MB = jnp.array(100000.0)
P0 = P1000MB
PSAT = jnp.array(610.78)

P400 = jnp.array(40000.0)
PHITP = jnp.array(15000.0)
PLBTM = jnp.array(105000.0)
PLOMD = jnp.array(64200.0)
PMDHI = jnp.array(35000.0)

G = jnp.array(9.81)
ROVG = R_D / G
EOMEG = jnp.array(7.2921e-5)
RERADIUS = jnp.array(1.0 / 6370.0)

DEGRAD = jnp.array(3.1415926 / 180.0)
DPD = jnp.array(360.0 / 365.0)
PI1 = jnp.array(3.1415926)
PI2 = jnp.array(2.0 * 3.1415926)

SVP1 = jnp.array(0.6112)
SVP2 = jnp.array(17.67)
SVP3 = jnp.array(29.65)
SVPT0 = jnp.array(273.15)

A2 = jnp.array(17.2693882)
A3 = jnp.array(273.16)
A4 = jnp.array(35.86)
PQ0 = jnp.array(379.90516)

PRANDTL = jnp.array(1.0 / 3.0)
W_ALPHA = jnp.array(0.3)
W_BETA = jnp.array(1.0)
FCDIF = jnp.array(1.0 / 3.0)
FCM = jnp.array(0.00003)
ASSELIN = jnp.array(0.025)

Z0LAND = jnp.array(0.10)
Z0SEA = jnp.array(0.001)
Z0MAX = jnp.array(0.008)

EPSQ = jnp.array(1.0e-12)
EPSQ2 = jnp.array(0.2)
CLIMIT = jnp.array(1.0e-20)
DEFC = jnp.array(0.0)
DEFM = jnp.array(99999.0)

N_CCN0 = jnp.array(1.0e8)
CB = jnp.array(25.0)
RHCRIT_LAND = jnp.array(0.75)
RHCRIT_SEA = jnp.array(0.80)

CM1 = jnp.array(2937.4)
CM2 = jnp.array(4.9283)
CM3 = jnp.array(23.5518)

EPSFC = jnp.array(1.0 / 1.05)
EPSWET = jnp.array(0.0)
RFCP = jnp.array(0.25) / CP
SCQ2 = jnp.array(50.0)
Q2INI = jnp.array(0.50)

WA = jnp.array(0.15)
WGHT = jnp.array(0.35)
WPC = jnp.array(0.075)
RLX = jnp.array(0.90)
RLAG = jnp.array(14.8125)

SLOPHT = jnp.array(0.001)
TLC = 2.0 * 0.703972477

BARE_USGS = 19
BARE_IGBP = 16

C2 = CP * RCV
GMA = -R_D * (1.0 - RCP) * 0.5
MWDRY = jnp.array(28.966)

XLV0 = jnp.array(3.15e6)
XLV1 = jnp.array(2370.0)
XLS0 = jnp.array(2.905e6)
XLS1 = jnp.array(259.532)
