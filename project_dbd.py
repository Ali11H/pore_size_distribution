# Importing needed libraries
import math
import numpy as np
import scipy
import pandas as pd
from numpy import interp, matmul
from scipy.optimize import nnls
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# defining the line-space function
def line_space(a, b, num):
    if num < 2:
        return b
    diff = (float(b) - a) / (num - 1)
    return [diff * i + a for i in range(num)]


# defining function evaluate
def evaluate_t_f(t):
    # model parameters from horizontal surface adsorption
    K = 0.0342
    m = 2.62
    F0 = 9.97
    lam = 0.216
    # getting F
    F = (K / pow(t, m)) + F0 * math.exp(-t / lam)
    # getting DF
    DF = ((-m * K) / pow(t, m + 1)) - (F0 / lam) * math.exp(-t / lam)
    return F, DF


# defining the main function evaluate_TCE

def evaluate_tce(P, R):
    la = 0.478
    precision = pow(10, -6)
    N = int(math.log(R / precision) / math.log(2))
    t_min_1 = 0
    t_max_1 = R
    # determining the condensation temperature and pressure
    for n in range(0, N):

        tm = 0.5 * (t_max_1 + t_min_1)
        L1 = evaluate_t_f(tm)
        F1 = L1[1]
        F2 = -la / pow(R - tm, 2)
        if F1 > F2:
            t_max_1 = tm
        else:
            t_min_1 = tm
    tc = 0.5 * (t_max_1 + t_min_1)
    L_c = evaluate_t_f(tc)
    f_c = L_c[0]
    pc = math.exp(-(la / (R - tc)) - f_c)

    def int_f(t):
        x = line_space(t, R, 40)
        FX = []
        Y_t_i = []
        for iti in range(0, len(x)):
            L0 = evaluate_t_f(x[iti])
            FX.append(L0[0])
            y = FX[iti] * (R - x[iti])
            Y_t_i.append(y)
            iti += 1
        Y_i = np.array(Y_t_i)
        H_ax = np.array(x)
        area = scipy.integrate.trapezoid(Y_i, H_ax)
        val_f = 2 * (area / pow(R - t, 2))
        return val_f

    t_min_2 = 0
    t_max_2 = R
    # determining evaporation pressure and temperature
    for n in range(N):
        tm_2 = 0.5 * (t_max_2 + t_min_2)
        L2 = evaluate_t_f(tm_2)
        F1 = L2[0] - (la / (R - tm_2))
        F2 = int_f(tm_2)
        if F1 > F2:
            t_min_2 = tm_2
        else:
            t_max_2 = tm_2

    te = 0.5 * (t_max_2 + t_min_2)
    L_e = evaluate_t_f(te)
    f_e = L_e[0]
    pe = math.exp(-(la / (R - te)) - f_e)

    # Find t for all pressures

    tR = np.zeros(len(P))
    for ip in range(0, len(P)):
        if P[ip] > pc:
            tR[ip] = R
        else:
            lnp = math.log(P[ip])
            t_max_3 = tc
            t_min_3 = 0.1
            for n in range(N):
                tm_3 = 0.5 * (t_max_3 + t_min_3)
                val = -(la / (R - tm_3)) - lnp
                L3 = evaluate_t_f(tm_3)
                F3 = L3[0]
                if val > F3:
                    t_max_3 = tm_3
                else:
                    t_min_3 = tm_3

            tR[ip] = 0.5 * (t_min_3 + t_max_3)
    Tad = np.column_stack((tR, tR))
    # returning the array of t
    index_ad = np.argmax(P)
    for ip in range(0, len(P)):
        if P[ip] >= pe and ip > index_ad:
            Tad[ip, 1] = R

    return Tad, pc, pe


# defining the main function determining the pore size distribution


def eval_bdb_distribution(PV):
    # creating the sub data frames
    ads = PV.copy(deep=True)
    des = PV.copy(deep=True)
    ads_des = PV.copy(deep=True)
    # indexing the adsorption and desorption

    index_ads = int(ads[['Pressure(p/p0)']].idxmax())
    # separating the adsorption and desorption

    ads = ads.iloc[0:index_ads, :]
    des = des.iloc[index_ads + 1:len(des), :]
    # resetting indexes
    ads = ads.reset_index(drop=True)
    des = des.reset_index(drop=True)
    P_ads = ads.iloc[:, 0]
    V_ads = ads.iloc[:, 1]
    P_des = des.iloc[:, 0]
    V_des = des.iloc[:, 1]
    P = ads_des.iloc[:, 0]
    V = ads_des.iloc[:, 1]
    # creating an array of different pore sizes
    NR = 100
    RR = line_space(1, 30, NR)
    Pc = np.zeros(NR)
    arr_04 = [0.4]

    for r in range(NR):
        res = evaluate_tce(arr_04, RR[r])
        res_pc = float(res[1])
        Pc[r] = res_pc
    Nd = 50
    dd = interp(line_space(Pc[0], Pc[-1], Nd), Pc, RR) * 2
    dd = np.append(0, dd)
    Nd += 1
    tt = np.zeros((len(P), 1))
    vv = np.zeros((len(P), Nd))
    vv[:, 0] = np.ones(len(P))

    for n in range(1, Nd):
        result_00 = evaluate_tce(P, dd[n] / 2)
        tad = result_00[0]
        tads_arr_re = tad
        # adding adsorption and desorption
        for indie_l in range(len(tads_arr_re)):
            if indie_l > index_ads:
                tt[indie_l] = tads_arr_re[indie_l][1]
            else:
                tt[indie_l] = tads_arr_re[indie_l][0]

        for iti in range(len(P)):
            vv[iti, n] = 1 - pow(((0.5 * dd[n] - tt[iti]) / (0.5 * dd[n])), 2)

    # fit adsorption---------------------------------------------------------------------
    vv_f = vv[0:index_ads, :]
    # actual measured values
    V_f = V_ads
    # least square fitting
    ww = nnls(vv_f, V_f, maxiter=10000)
    ww_arr = ww[0]
    # fitted value
    V_fit = matmul(vv_f, ww_arr)
    err_0 = 0
    for n in range(len(V_fit)):
        err_0 += pow((V_fit[n] - V_f[n]), 2)

    def entropy(x):
        sum_sq = sum(i ** 2 for i in x)
        return sum_sq

    # defining the tolerance
    ep = 5 * pow(10, -2)

    # defining error_ads/ function for non-linear constraint on error

    def error_ads(x):
        error_1 = ep - abs(np.sum(pow(np.subtract(matmul(vv_f, x), V_f), 2)) - 1.1 * err_0)
        return error_1
    # building the constraints(non-linear)
    non_lin_cons = {'type': 'ineq', 'fun': error_ads}
    # bounds, to insure positive values
    bonds = [(0, None)]
    # initial guess is ww_arr
    x_init = ww_arr
    # launching minimization
    ww1 = minimize(entropy, x0=x_init, constraints=non_lin_cons, bounds=bonds,
                   options={'disp': True})
    # storing the result
    psd = np.array(ww1.x)
    # fit desorption---------------------------------------------------------------------
    vv_d = vv[(index_ads + 1):len(P), :]
    # actual measured values
    V_f_des = V_des
    # least square fitting
    ww_1 = nnls(vv_d, V_f_des, maxiter=10000)
    ww_arr_1 = ww_1[0]
    # fitted value
    V_fit_des = matmul(vv_d, ww_arr_1)
    # defining the error
    err_des = 0
    for n in range(len(V_fit_des)):
        err_des += pow((V_fit_des[n] - V_f_des[n]), 2)

    # defining error_des/ function for non-linear constraint on error

    def error_des(x):
        error_2 = ep - abs(np.sum(pow(np.subtract(matmul(vv_d, x), V_f_des), 2)) - 1.1 * err_des)
        return error_2
    # building the constraints(non-linear)
    non_lin_cons2 = {'type': 'ineq', 'fun': error_des}
    # bounds, to insure positive values
    bonds2 = [(0, None)]
    # initial guess is ww_arr
    x_init1 = ww_arr_1
    # x_init = np.zeros(len(ww_arr))
    ww2 = minimize(entropy, x0=x_init1, constraints=non_lin_cons2, bounds=bonds2,
                   options={'disp': True})
    # storing the result
    psd2 = np.array(ww2.x)
    # fit adsorption and desorption---------------------------------------------------------------------
    vv_ads_des = vv[0:len(P), :]
    # actual measured values
    V_f_ads_des = V
    # least square fitting
    ww_2 = nnls(vv_ads_des, V_f_ads_des, maxiter=10000)
    ww_arr_2 = ww_2[0]
    # fitted value
    V_fit_ads_des = matmul(vv_ads_des, ww_arr_2)
    # defining the error
    err_ads_des = 0
    for n in range(len(V_fit_ads_des)):
        err_ads_des += pow((V_fit_ads_des[n] - V_f_ads_des[n]), 2)

    # defining error_ads_des/ function for non-linear constraint on error
    def error_ads_des(x):
        error_3 = ep - abs(np.sum(pow(np.subtract(matmul(vv_ads_des, x), V_f_ads_des), 2)) - 1.1 * err_ads_des)
        return error_3

    # building the constraints(non-linear)
    non_lin_cons2 = {'type': 'ineq', 'fun': error_ads_des}
    # bounds, to insure positive values
    bonds3 = [(0, None)]
    # initial guess is ww_arr
    x_init2 = ww_arr_2
    # x_init = np.zeros(len(ww_arr))
    ww3 = minimize(entropy, x0=x_init2, constraints=non_lin_cons2, bounds=bonds3,
                   options={'disp': True})
    # storing the result
    psd3 = np.array(ww3.x)
    # PLOTTING THE RESULTS----------------------------------------------------
    # plotting the fitting
    plt.subplot(1, 2, 1)
    # plotting the fit adsorption
    plt.plot(P_ads, V_fit, color='r', label='fit_ads')
    # plotting the fit desorption
    plt.plot(P_des, V_fit_des, color='k', label='fit_des')
    # plotting the fit adsorption desorption
    plt.plot(P, V_fit_ads_des, color='b', label='fit_ads_des')
    # measured ads/des
    plt.scatter(P, V, color='c', marker="^", label='measured')
    plt.xlabel('Pressure(p/P0)')
    plt.ylabel('V_ads')
    plt.title("adsorption isotherm")
    plt.legend()
    # plotting the PSD
    plt.subplot(1, 2, 2)
    plt.plot(dd, psd, color='r', label='psd_ads')
    plt.plot(dd, psd2, color='k', label='psd_des')
    plt.plot(dd, psd3, color='b', label='psd_ads_des')
    # plt.xscale('log')
    plt.xlabel('pore Diameter(nm)')
    plt.ylabel('volume')
    plt.title("PSD")
    plt.legend()
    plt.show()
    return P, V, V_fit, V_f_des, V_fit_ads_des, psd, psd2, psd3


# testing the function


# importing some data


# reading the data
col_names = ['Pressure(p/p0)', 'V_ads']
read_file = pd.read_csv(r'write-the file directory\ALIHA_S2F.txt', sep='\s+',
                        names=col_names)
# creating the sub data frames
adsorption = read_file.copy(deep=True)
desorption = read_file.copy(deep=True)
full = read_file.copy(deep=True)

eval_bdb_distribution(full)
