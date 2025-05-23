import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

svensson = []
res = []
with open('./g.svensson.dat','r') as f:
    lines = f.readlines()
    res.append(lines)
    for d in res:
        for p in d:
            if "#" in p:
                continue
            s = p.split()
            svensson.append([float(s[0]),float(s[1])])

r_svensson = [x[0] for x in svensson]
g_svensson = [x[1] for x in svensson]

# He-liquid measured rhos
meas_file = './rhos_per_rho_donnelly.dat'
with open(meas_file,'r') as f:
    lines = f.readlines()

meas=[]
for line in lines:
    if "#" in line: continue
    s = line.split()
    dat = []
    for d in s:
        dat.append(float(d))
    try:
        meas.append([dat[0],dat[1]])
    except:
        pass

T_meas,rhos_meas = zip(*meas)


#
# boninsegni_permutation_sampling.condmat205.pdf
# PIMC data
#
# Boninsegni
# Temp    rho    N     T             V  
Bondat = [[1.1765, 0.02182, 544, 14.123, 0.028, -21.3127, 0.0025],
          [1.379,  0.02182, 466, 14.201, 0.024, -21.3195, 0.0029],
          [1.600,  0.02183, 400, 14.334, 0.034, -21.3435, 0.0037],
          [1.818,  0.02186, 352, 14.468, 0.059, -21.3913, 0.0060],
          [2.000,  0.02191, 320, 14.862, 0.071, -21.4810, 0.0078],
          [2.353,  0.02191, 272, 15.821, 0.062, -21.5791, 0.0072]]

T_bon =   [x[0] for x in Bondat]
E_bon = [x[3]+x[5] for x in Bondat]
V_bon = [x[5] for x in Bondat]



#
# Ceperley Pollock PRL 1986 
# Temp    rho    <V>    <T>        
CPdat = [[4.0,      .01932,  -18.91,  15.65],
           [3.333,    .02072,  -20.38,  16.00], 
           [2.857,    .02142,  -21.14,  15.99],
           [2.50,        .02179,  -21.52,  15.90 ],
           [2.353,       .02191,  -21.60,  15.75 ],
           [2.222,       .02197,  -21.70,  15.89 ],
           [2.105,       .02194,  -21.57,  15.10 ],
           [ 2.0  ,      .02191,  -21.57,  15.05 ],
           [ 1.818,      .02186,  -21.44,  14.71 ],
           [ 1.600,      .02183,  -21.39,  14.40 ],
           [ 1.379,      .02182,  -21.35,  14.23 ],
           [ 1.1765,     .02182,  -21.35,  14.17 ]]

T_cp =   [x[0] for x in CPdat]
E_cp = [x[2]+x[3] for x in CPdat]
V_cp = [x[2] for x in CPdat]



#
# exp energy data (Integrated from CV, added -7.175)
#
Expdat = []
res = []
with open('./E_He4_energy_from_CV.dat','r') as f:
    lines = f.readlines()
    res.append(lines)
    for d in res:
        for p in d:
            s = p.split()
            Expdat.append([float(s[0]),float(s[1])])

T_exp = [x[0] for x in Expdat]
E_exp = [x[1] for x in Expdat]




# Load the HDF5 file

def load_hdf5(filename):    
    with h5py.File(filename, "r") as f:
        try:
            # Energies
            E_th = f["E_th/E"][()]
            std_E_th = f["E_th/std_E"][()]
            E_vir = f["E_vir/E"][()]
            std_E_vir = f["E_vir/std_E"][()]
            V = f["E_vir/V"][()]
            std_V = f["E_vir/std_V"][()]
            
            # superfluid fraction
            try:
                rho_s = f["superfluid_fraction/rhos"][()]
                std_rho_s = f["superfluid_fraction/std"][()]
            except:
                pass
            
            # g(r)
            try:
                gs = f["radial_distribution/g"][:]
                rs = f["radial_distribution/r"][:]
            except:
                pass
            
            ok = True
        except:
            print("no required data in HDF5 file (yet)")
            ok = False

        # Metadata
        date = f["metadata/date"][()].decode()
        
        
        T = f["metadata/T"][()]
        N = f["metadata/N"][()]
        tau = f["metadata/tau"][()]
        action = f["metadata/action"][()].decode()
        print('hdf5 ',filename)
        print(f"Result Date: {date}")
    if not ok:
        return False
        
    return E_th, std_E_th, E_vir, std_E_vir, rho_s, std_rho_s, gs, rs, T, tau, action, V, std_V, N


filemarkers = ['results.He_liquid_chin_']


def readdat(ids):
    res = []    
    for file in os.listdir('./'):
        for filemarker in filemarkers:
            if file.startswith(filemarker):
                ok = True
                for id in ids:
                    if not id in file: ok = False
                
                if not ok: continue
                rr = load_hdf5(file)
                if rr==False:
                    continue
                res.append(rr)
    return res

def plot_E(Edata, col, action_str, N, first):
    plt.figure(1)
    if first:
        plt.clf()
    #plt.plot(T_bon,E_bon,'bo-',markersize=4, label='Boninsegni')
    #plt.plot(T_cp,E_cp,'rx',markersize=4, label='Ceperley-Pollock')
    if first:
        plt.plot(T_exp,E_exp,'b-', label='Experiment')
    
    lab="Virial estimator"+" N="+str(N)
    Edata.sort()
    Ts = [d[0] for d in Edata]
    Es = [d[1] for d in Edata]
    stds = [d[2] for d in Edata]
    plt.errorbar(Ts,Es,stds,marker='o',color=col,ls='none',label=lab,markersize=4,capsize=3)

    plt.xlabel("T [K]")
    plt.ylabel("E [K]")
    plt.title("Energy")
    plt.legend()
    plt.pause(0.001)
    plt.draw()

    
def plot_g(gdata, col, action_str, N, first):    
    # Plot g(r)
    
    plt.figure(2)
    if first:
        plt.clf()
    if first:
        plt.plot(r_svensson, g_svensson,'bo', label="Exp. Neutron diffraction at T=1.0 K", markersize=3)

    for dat in gdata:
        T, tau, rs, gs = dat
        if T != 1.0:
            continue
        tauscr = f'{tau:.3f}'
        Tscr = f'{T:.3f}'
        plt.plot(rs, gs, color=col, label = "T = "+Tscr+" N="+str(N))
    plt.xlabel("r [Å]")
    plt.ylabel("g(r)")
    plt.title("Radial Distribution Function")
    plt.ylim([0,2.1])
    plt.tight_layout()
    plt.legend()    
    plt.pause(0.001)
    plt.draw()
    
def plot_rhos(rhosdata, col, action_str, N, first):
    plt.figure(3)
    if first:
        plt.clf()
    if first:
        plt.plot(T_meas,rhos_meas,'b-',label=r'Exp. He$^4$ superfluid fraction')
    lab = " N="+str(N)
    rhosdata.sort()
    Ts = [d[0] for d in rhosdata]
    rhoss = [d[1] for d in rhosdata]
    stds = [d[2] for d in rhosdata]
    #plt.plot(Ts, rhoss, '-', color=col)
    plt.errorbar(Ts,rhoss,stds, marker='o',color=col,ls='none',label=lab,markersize=4,capsize=3)

    plt.xlim([0.0,5.1])
    plt.ylim([-0.1,1.2])
    plt.title("Superfluid fraction")
    plt.hlines(0,0,5,linestyles='dashed')
    plt.xlabel("T [K]")
    plt.ylabel(r"$\rho_s/\rho$")
    plt.legend()
    plt.pause(0.001)
    plt.draw()


  
def plot_V(Vdata, col, action_str, N, first):
    plt.figure(4)                               
    if first:
        plt.clf()
    if first:
        plt.plot(T_bon,V_bon,'bo-',markersize=4, label='PIMC: Boninsegni et al.')
        plt.plot(T_cp,V_cp,'rx-',markersize=4, label='PIMC: Ceperley & Pollock')

    lab = " N="+str(N)
    Vdata.sort()
    Ts = [d[0] for d in Vdata]
    Vs = [d[1] for d in Vdata]
    stds = [d[2] for d in Vdata]
    #plt.plot(Ts, Vs, '-', color=col)
    plt.errorbar(Ts,Vs,stds, marker='o',color=col,ls='none',label=lab,markersize=4,capsize=3)
    
    
    plt.xlabel("T [K]")
    plt.ylabel("V [K]")
    plt.title("Potential Energy")
    plt.legend()
    plt.pause(0.001)
    plt.draw()

def plot():
    

    #example filename = "results.He_liquid_chin_T4.0_tau0.1_M9_N16_bose.dat.h5"
    idlist  = [ ['tau0.01','16','olive'], ['tau0.01','64','purple']]


    
    ok = False
    first = True
    for ids in idlist:
        res = readdat([ids[0],ids[1]])
        col = ids[2] 
        if len(res)==0:
            print("no res, skip")
            continue
        Edata = []
        Vdata = []
        gdata = []
        rhosdata = []
        for r in res:
            d = [r[i] for i in [8,2,3]] # virial estimator
            Edata.append(d)
            print('Energies: ',d)
            d = [r[i] for i in [8,11,12]]
            Vdata.append(d)
            
            d = [r[i] for i in [8,9,7,6]]
            gdata.append(d)
            
            d = [r[i] for i in [8,4,5]]
            rhosdata.append(d)
            action = r[10]
            N = r[13]
            ok = True
    
            # E_th, std_E_th, E_vir, std_E_vir, rho_s, std_rho_s, gs, rs, T, tau, action, V, std_V, N

        if not ok:
            print("no data to plot")
    
        if action=="chin":
            action_str = "Chin Action" 
        else:
            action_str = "Primitive Action"
        plot_E(Edata, col, action_str, N, first)
        plot_V(Vdata, col, action_str, N, first)
        plot_g(gdata, col, action_str, N, first)
        plot_rhos(rhosdata, col, action_str, N, first)
        first = False

def onclick(event):
    if event.button == 1:
        fig.clf()
        plot()
    

if __name__=='__main__':
    fig = plt.figure(1)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plot()
    print('CLICK FIGURE 1 CANVAS TO RE-READ DATA')
    plt.show()
