import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode


# =============================================================================
# Equations
# =============================================================================
''' Rate equations of the reporter in its 4 states: unphosphorilated on the cytosol(rup), 
unphosphorilated on the nucleus (rnu), phosphorilated on the cytosol (rcp), phosphorilated on the nucleus(rnp)

The equations had been taken from the supplementary material of Regot 2014 paper
'''

def RCU(t,R, params): # el return de esta ecuacion es drcu//dt
    [r_total, kv, kiu, keu, kip, kep, kcat, Km ,kdc,kdn ,Kmd ,amp,w] = params
    [Kin_n,Kin_c] = KIN(t,amp,w) 
    [rcu,rnu,rcp,rnp]= R
    return -Kin_c * kcat * (rcu/(rcu + Km)) + kdc  * (rcp/(rcp + Kmd)) -kiu * rcu + keu * rnu

def RNU(t,R, params): # el return de esta ecuacion es drnu//dt
    [r_total, kv, kiu, keu, kip, kep, kcat, Km ,kdc,kdn ,Kmd ,amp,w] = params
    [Kin_n,Kin_c] = KIN(t,amp,w)
    [rcu,rnu,rcp,rnp]= R
    return -Kin_n * kcat * (rnu/(rnu + Km)) + kdn  * (rnp/(rnp + Kmd)) + kv * kiu * rcu - kv* keu * rnu

def RCP(t,R, params): # el return de esta ecuacion es drcp//dt
    [r_total, kv, kiu, keu, kip, kep, kcat, Km ,kdc,kdn ,Kmd ,amp,w] = params
    [Kin_n,Kin_c] = KIN(t,amp,w)  
    [rcu,rnu,rcp,rnp]= R
    return Kin_c * kcat * (rcu/(rcu + Km)) - kdc  * (rcp/(rcp + Kmd))  - kip * rcp + kep * rnp

def RNP(t,R, params): # el return de esta ecuacion es drcp//dt
    [r_total, kv, kiu, keu, kip, kep, kcat, Km ,kdc,kdn ,Kmd ,amp,w] = params
    [Kin_n,Kin_c] = KIN(t,amp,w)  
    [rcu,rnu,rcp,rnp]= R
    return Kin_n * kcat * (rnu/(rnu + Km)) - kdn  * (rnp/(rnp + Kmd)) + kv * kip * rcp -kv * kep * rnp


# =============================================================================
# Kinase activity
# =============================================================================
def KIN(t,amp,w):
    Kin_c = 0.3 * amp * (1+ np.cos(w*t))
    Kin_n = 0.7* amp * (1+ np.cos(w*t))
    return [Kin_n,Kin_c]

#def KIN(t,amp,w):
#    i = 1 if t > 20 else  0
#    Kin_c = 0.3 * amp * i
#    Kin_n = 0.7* amp * i    
#    return [Kin_n,Kin_c]

#
#def KIN(t,amp,w):  
#    Kin_c = 0.3 * amp * np.random.uniform(1,2)
#    Kin_n = 0.7 * amp * np.random.uniform(1,2)
#    return [Kin_n,Kin_c]
# =============================================================================
# main function
# =============================================================================    
def f(t, R, params): #esta funcion une todas las anteriores
    return [RCU(t,R, params), RNU(t,R, params),RCP(t,R, params),RNP(t,R, params)]


def split_R(R): # esta funcion la uso para que me devuelva el resultado de cada variable en una lista separada.
    su = [[]]*len(R[0])
    for n in range(len(R[0])):
        su[n] = list(map(lambda x: x[n], R))
    return su

#%%
def integrate(R0, params, T, steps):
    
    integr = ode(f).set_integrator('vode', method='adams', order=10, atol=1e-6,with_jacobian=False).set_f_params(params)
    integr.set_initial_value(R0, 0)
    
    dt = T/steps
    R = [R0]; t = [0]
   # n = 0
    while integr.successful() and integr.t <= T:
        #n += 1    
        integr.integrate(integr.t + dt)
        R.append(integr.y);  t.append(integr.t)
    return t, split_R(R)

#%%

# =============================================================================
# Parameters
# =============================================================================
'''
units: micromolar & minutes
'''

r_total = 0.4
kv = 4
kiu = 0.44
keu = 0.11
kip = 0.16
kep = 0.2
kcat = 20
Km = 3
kdc = 0.03
kdn = 0.03
Kmd = 0.1


amp = 1
w = 1
params = [r_total, kv, kiu, keu, kip, kep, kcat, Km ,kdc,kdn ,Kmd ,amp,w]


# =============================================================================
# Initial conditions
# =============================================================================


R0 = [0.4,0,0,0]
T = 300
steps = 100000

t, [rcu,rnu,rcp,rnp] = integrate(R0, params, T, steps)

#plt.figure(figsize = (20,10))
#plt.plot(t, [KIN(i,amp,w)[0] for i in t], label = 'Kin_n')
#plt.plot(t, rnp, label = 'rnp')
#plt.legend()
#plt.plot([a+b+c/kv+d/kv for a,b,c,d in zip(rcu, rcp,rnp,rnu)])

#%%

amplitude = [0.05,0.1,1]
omega = [2 *np.pi /7/4, 2 *np.pi /7/2, 2 *np.pi /7/1, 2 *np.pi /7/0.5]
save_path_name = '/home/fabris/Documents/Dyncode/sensor_simulations/figures/'


##############################################################################
### Plotting parameters
###############################################################################    
plt.rcdefaults(); #xlim = [-5,T_+5] ; ylim = [-1.1,1.1] ;         
Cols = len(amplitude)
Rows = len(omega)


###############################################################################
### Figure
###############################################################################    

fig, axs = plt.subplots(Rows, Cols, sharex=True, sharey=True, figsize=(8.27*3, 11.69))
fig.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.8, wspace=0.1, hspace=0.1)

for row,w in  enumerate(omega):
    text_w = r'$\omega = $'+str(w/(2*np.pi/7))+r'$  \frac{2\pi}{7 min}$' 
    axs[row,Cols-1].text(1.2,0.5 ,text_w, ha='center', va='center', transform=axs[row,-1].transAxes, fontsize=10)
    for col,amp in enumerate(amplitude):
        text_amp = str(amp) + r'$\mu M$' 
        axs[0,col].text(0.5,1.2, text_amp, ha='center', va='center', transform=axs[0,col].transAxes, fontsize=10)
        ax = axs[row,col]; ax.grid(False);

    ################################################
    #### download data
    ################################################
        params = [r_total, kv, kiu, keu, kip, kep, kcat, Km ,kdc,kdn ,Kmd ,amp,w]
        t, [rcu,rnu,rcp,rnp] = integrate(R0, params, T, steps)
    
        ################################################
        #### Plotting
        ################################################
        ax.plot(t, [KIN(i,amp,w)[0] for i in t], label = 'nuclear kinase',linewidth = 0.8)
        ax.plot(t, [-sum(x)+0.5 for x in zip(rnu, rnp)] , label = 'nuclear p+u reporter ',linewidth = 0.8)
        #ax.plot(t, rnp, label = 'nuclear p reporter ',linewidth = 0.8)
        #ax.plot(t, rcu, label = 'cytosol u reporter ',linewidth = 0.8)
        #ax.plot(t, rcp, label = 'cytosol p reporter ',linewidth = 0.8)


        #ax.plot(t, [x+y for x,y in zip(rnp,rnu)], label = 'nuclear reporter (p+u)',linewidth = 0.8)
        
        #ax.set_ylim(ylim);
        #ax.set_xlim(xlim)
        
        if row == Rows - 1:
            ax.set_xlabel('time (min)', fontsize=15)
            ax.xaxis.set_label_coords(0.5, -0.15);

        if col == 0:
            ax.set_ylabel('Amplitude ' + r'$(\mu M)$', fontsize=15);
            ax.yaxis.set_label_coords(-0.15, 0.5)

axs[0,Cols-1].legend(loc='upper right', frameon=False, bbox_to_anchor=(1.6, 1.4))

plt.savefig(save_path_name + 'Sensor8.pdf', format='pdf')

#%% Esto es para calcular la FFT


def norm_fft(data, Dt, max_freq = None):
    '''
    FFT for each time serie. Returns the FFT on a orthonomal basis so Parseval equality is satisfied. 
    Inputs:
        - data: amplitude time serie
        - Dt: sampling rate
        - max_freq (actually is not working): maximum frequency of the FFT 
    Returns:
        - yf (complex ndarray): FFT transform time series
        - xf: A list of the transformed angular frequencies
    '''
    N = data.shape[0]
    Nf = N // 2 #if max_freq is None else int(max_freq * T)
    xf = np.linspace(0.0, 0.5 / Dt, N // 2)
    yf =  np.fft.fft(data,norm='ortho')
    return xf[:Nf]*(2*np.pi), yf[:Nf]


###############################################################################
### Figure
###############################################################################    

fig, axs = plt.subplots(Rows, Cols, sharex=True, sharey=True, figsize=(8.27*3, 11.69))
fig.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.8, wspace=0.1, hspace=0.1)

for row,w in  enumerate(omega):
    text_w = r'$\omega = $'+str(w/(2*np.pi/7))+r'$  \frac{2\pi}{7 min}$' 
    #axs[row,Cols-1].text(1.2,0.5 ,text_w, ha='center', va='center', transform=axs[row,-1].transAxes, fontsize=10)
    for col,amp in enumerate(amplitude):
        text_amp = str(amp) + r'$\mu M$' 
        axs[0,col].text(0.5,1.2, text_amp, ha='center', va='center', transform=axs[0,col].transAxes, fontsize=10)
        ax = axs[row,col]; ax.grid(False);

    ################################################
    #### download data
    ################################################
        params = [r_total, kv, kiu, keu, kip, kep, kcat, Km ,kdc,kdn ,Kmd ,amp,w]
        t, [rcu,rnu,rcp,rnp] = integrate(R0, params, T, steps)
        
    ################################################
    #### FFT
    ################################################
        Kn_xf,Kn_yf =  norm_fft(np.array([KIN(i,amp,w)[0] for i in t]), np.diff(t)[0])        
        rn_xf,rn_yf =  norm_fft(np.array([sum(x) for x in zip(rnu, rnp)]), np.diff(t)[0])
        
        ################################################
        #### Plotting
        ################################################
        ax.plot(Kn_xf, np.abs([ i*j for i,j in zip(Kn_xf,Kn_yf)])**2, label = 'nuclear kinase',linewidth = 0.8,markersize=0.5)
        ax.plot(rn_xf, np.abs([i*j for i,j in zip(rn_xf,rn_yf)])**2 , label = 'nuclear p+u reporter ',linewidth = 1.5)
        #ax.axhline(1/(2*np.pi),color='black',linestyle=':')
        
        if row == Rows - 1:
            ax.set_xlabel('angular frequency (min)', fontsize=15)
            ax.xaxis.set_label_coords(0.5, -0.15);

        if col == 0:
            ax.set_ylabel('Power specturm  ' + r'$(\mu M)$', fontsize=15);
            ax.yaxis.set_label_coords(-0.15, 0.5)
        ax.set_ylim([-0.1,1])

axs[0,Cols-1].legend(loc='upper right', frameon=False, bbox_to_anchor=(1.6, 1.4))

plt.savefig(save_path_name + 'FFT_Sensor_7.pdf', format='pdf')

