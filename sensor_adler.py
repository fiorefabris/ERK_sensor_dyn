import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from math import ceil
from scipy.interpolate import interp1d



# =============================================================================
# Kinase activity
# =============================================================================


# Time evolution funtion 
def time_evolution_(omega,alpha,D,dt = 0.0001 ,T = 10000, d=1):
    ''' Simulated data of the Adler phase model with gaussian white noise. 
    
    ------------------------------------------------------------
    INPUTS:
        - omega,alpha (real numbers): parameters of the Adler deterministic
         equation
        - D (real number): noise strengh
        - dt (positive real number, default dt=0.0001): time steps of
        the simulation
        - T (positive real number > dt, default T = 10000): Total time 
        of the simulation
        - d (integer number, default d = 1): decimation factor

    ------------------------------------------------------------
    OUTPUTS:
        - theta : phase simulated variables

'''

    n     = int(T/dt) # total number of steps
    
    #variables
    theta = np.zeros(ceil(n/d))
    t     = np.zeros(ceil(n/d))  

    #### Initial conditions ############################
    ####################################################
    np.random.seed()
    theta_past = np.random.uniform(0,2)*np.pi
    
    #### Time evolution ################################
    ####################################################

    for i in range(n-1):
        u = np.sqrt(-2* np.log(np.random.uniform(0,1))) * np.cos(2*np.pi*np.random.uniform(0,1))
        k = dt * (omega + alpha * np.sin(theta_past))
        l = np.sqrt(dt * 2 * D) * u

        theta_present = theta_past + dt/2 * (2*omega + alpha * (np.sin(theta_past) + np.sin(theta_past + l + k) )) + np.sqrt(dt * 2 * D) * u
        theta[i] = theta_past;theta_past = theta_present
        if i != 0: t [i] = t[i-1] + dt

    return(t,theta)

# =============================================================================
# Equations
# =============================================================================
''' Rate equations of the reporter in its 4 states: unphosphorilated on the cytosol(rup), 
unphosphorilated on the nucleus (rnu), phosphorilated on the cytosol (rcp), phosphorilated on the nucleus(rnp)

The equations had been taken from the supplementary material of Regot 2014 paper
'''

def RCU(t,R, params): # el return de esta ecuacion es drcu//dt
    [r_total, kv, kiu, keu, kip, kep, kcat, Km ,kdc,kdn ,Kmd ,amp, adler_] = params
    [Kin_n,Kin_c] = KIN(t,adler_)
    [rcu,rnu,rcp,rnp]= R
    return -Kin_c * kcat * (rcu/(rcu + Km)) + kdc  * (rcp/(rcp + Kmd)) -kiu * rcu + keu * rnu

def RNU(t,R, params): # el return de esta ecuacion es drnu//dt
    [r_total, kv, kiu, keu, kip, kep, kcat, Km ,kdc,kdn ,Kmd ,amp,adler_] = params
    [Kin_n,Kin_c] = KIN(t,adler_)
    [rcu,rnu,rcp,rnp]= R
    return -Kin_n * kcat * (rnu/(rnu + Km)) + kdn  * (rnp/(rnp + Kmd)) + kv * kiu * rcu - kv* keu * rnu

def RCP(t,R, params): # el return de esta ecuacion es drcp//dt
    [r_total, kv, kiu, keu, kip, kep, kcat, Km ,kdc,kdn ,Kmd ,amp,adler_] = params
    [Kin_n,Kin_c] = KIN(t,adler_) 
    [rcu,rnu,rcp,rnp]= R
    return Kin_c * kcat * (rcu/(rcu + Km)) - kdc  * (rcp/(rcp + Kmd))  - kip * rcp + kep * rnp

def RNP(t,R, params): # el return de esta ecuacion es drcp//dt
    [r_total, kv, kiu, keu, kip, kep, kcat, Km ,kdc,kdn ,Kmd ,amp,adler_] = params
    [Kin_n,Kin_c] = KIN(t,adler_)
    [rcu,rnu,rcp,rnp]= R
    return Kin_n * kcat * (rnu/(rnu + Km)) - kdn  * (rnp/(rnp + Kmd)) + kv * kip * rcp -kv * kep * rnp

# =============================================================================
# Kinase activity
# =============================================================================

def KIN(t,adler_):
    Kin_c = 0.3 * amp * (np.cos(float(adler_(t))) + 1)
    Kin_n = 0.7* amp * (np.cos(float(adler_(t))) +1)
    return [Kin_n,Kin_c]

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
   
    while integr.successful() and integr.t <= T:
        integr.integrate(integr.t + dt)
        R.append(integr.y);  t.append(integr.t)
    return t, split_R(R)


#%%

# =============================================================================
# Initial conditions
# =============================================================================


R0 = [0.4,0,0,0]
T = 300
steps = 10000
dt = T/steps

t, theta = time_evolution_(2* np.pi/7,1.01 * 2* np.pi/7 ,0.05,T/steps ,T*2, d=1)
adler_ = interp1d(t,theta)

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


#amp = 0.5
#params = [r_total, kv, kiu, keu, kip, kep, kcat, Km ,kdc,kdn ,Kmd ,
#          amp,adler_]




#t, [rcu,rnu,rcp,rnp] = integrate(R0, params, T, steps)

#plt.figure(figsize = (20,10))
#plt.plot(t, [KIN(i,adler_)[0] for i in t], label = 'Kin_n')
#plt.plot(t, rnp, label = 'rnp')
#plt.legend()
##plt.plot([a+b+c/kv+d/kv for a,b,c,d in zip(rcu, rcp,rnp,rnu)])

#%%
amplitude = [0.05,0.1,0.4,0.7]
save_path_name = '/home/fabris/Documents/Dyncode/sensor_simulations/figures/'


##############################################################################
### Plotting parameters
###############################################################################    
plt.rcdefaults(); #xlim = [-5,T_+5] ; ylim = [-1.1,1.1] ;         
Rows = len(amplitude)
Cols = 1


###############################################################################
### Figure
###############################################################################    

fig, axs = plt.subplots(Rows, Cols, sharex=True, sharey=True, figsize=(8.27*3, 11.69))
fig.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.8, wspace=0.1, hspace=0.1)

text_w = r'$\omega = \frac{2\pi}{7 min}$' + r' $\alpha = 1.01 \times \frac{2\pi}{7 min}$' + r' $ D = 0.05$'
axs[0].text(0.5,1.2,text_w, ha='center', va='center', transform=axs[0].transAxes, fontsize=10)


for row,amp in  enumerate(amplitude):
        t, theta = time_evolution_(2* np.pi/7,1.01 * 2* np.pi/7 ,0.05,T/steps ,T*2, d=1)
        adler_ = interp1d(t,theta)
        ax = axs[row]; ax.grid(False);
        if row == 0: ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.6, 1.4))
        text_amp = str(amp) + r'$\mu M$' 
        ax.text(1.2,0.5 , text_amp, ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ################################################
    #### download data
    ################################################
        params = [r_total, kv, kiu, keu, kip, kep, kcat, Km ,kdc,kdn ,Kmd, amp,adler_]
        t, [rcu,rnu,rcp,rnp] = integrate(R0, params, T, steps)
    
        ################################################
        #### Plotting
        ################################################
        ax.plot(t,[KIN(i,adler_)[0] for i in t], label = 'nuclear kinase',linewidth = 1)
        ax.plot(t, [-sum(x)+0.5 for x in zip(rnu, rnp)] , label = 'nuclear reporter (p+u) ',linewidth = 1)
        #ax.plot(t, rnp, label = 'nuclear p reporter ',linewidth = 0.8)
        #ax.plot(t, rcu, label = 'cytosol u reporter ',linewidth = 0.8)
        #ax.plot(t, rcp, label = 'cytosol p reporter ',linewidth = 0.8)


        #ax.plot(t, [x+y for x,y in zip(rnp,rnu)], label = 'nuclear reporter (p+u)',linewidth = 0.8)
        
        #ax.set_ylim(ylim);
        #ax.set_xlim(xlim)
        

ax.set_xlabel('time (min)', fontsize=15)
ax.xaxis.set_label_coords(0.5, -0.15);

ax.set_ylabel('Amplitude ' + r'$(\mu M)$', fontsize=15);
ax.yaxis.set_label_coords(-0.5, 0.5)


plt.savefig(save_path_name + 'Sensor5.pdf', format='pdf')


#%%

# =============================================================================
# Initial conditions
# =============================================================================


R0 = [0.4,0,0,0]
T = 300
steps = 10000
dt = T/steps

t, theta = time_evolution_(2* np.pi/7,0.95 * 2* np.pi/7 ,0.05,T/steps ,T*2, d=1)
adler_ = interp1d(t,theta)

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


#%%
amplitude = [0.05,0.1,0.4,0.7]
save_path_name = '/home/fabris/Documents/Dyncode/sensor_simulations/figures/'


##############################################################################
### Plotting parameters
###############################################################################    
plt.rcdefaults(); #xlim = [-5,T_+5] ; ylim = [-1.1,1.1] ;         
Rows = len(amplitude)
Cols = 1


###############################################################################
### Figure
###############################################################################    

fig, axs = plt.subplots(Rows, Cols, sharex=True, sharey=True, figsize=(8.27*3, 11.69))
fig.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.8, wspace=0.1, hspace=0.1)

text_w = r'$\omega = \frac{2\pi}{7 min}$' + r' $\alpha = 0.95 \times \frac{2\pi}{7 min}$' + r' $ D = 0.05$'
axs[0].text(0.5,1.2,text_w, ha='center', va='center', transform=axs[0].transAxes, fontsize=10)


for row,amp in  enumerate(amplitude):
        if row == 0: ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.6, 1.4))
        ax = axs[row]; ax.grid(False);
        text_amp = str(amp) + r'$\mu M$' 
        ax.text(1.2,0.5 , text_amp, ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ################################################
    #### download data
    ################################################
        params = [r_total, kv, kiu, keu, kip, kep, kcat, Km ,kdc,kdn ,Kmd, amp,adler_]
        t, [rcu,rnu,rcp,rnp] = integrate(R0, params, T, steps)
    
        ################################################
        #### Plotting
        ################################################
        ax.plot(t,[KIN(i,adler_)[0] for i in t], label = 'nuclear kinase',linewidth = 0.8)
        ax.plot(t, [sum(x) for x in zip(rnu, rnp)] , label = 'nuclear reporter (p+u) ',linewidth = 0.8)
        #ax.plot(t, rnp, label = 'nuclear p reporter ',linewidth = 0.8)
        #ax.plot(t, rcu, label = 'cytosol u reporter ',linewidth = 0.8)
        #ax.plot(t, rcp, label = 'cytosol p reporter ',linewidth = 0.8)


        #ax.plot(t, [x+y for x,y in zip(rnp,rnu)], label = 'nuclear reporter (p+u)',linewidth = 0.8)
        
        #ax.set_ylim(ylim);
        #ax.set_xlim(xlim)
        

ax.set_xlabel('time (min)', fontsize=15)
ax.xaxis.set_label_coords(0.5, -0.15);

ax.set_ylabel('Amplitude ' + r'$(\mu M)$', fontsize=15);
ax.yaxis.set_label_coords(-0.5, 0.5)


plt.savefig(save_path_name + 'Sensor6.pdf', format='pdf')