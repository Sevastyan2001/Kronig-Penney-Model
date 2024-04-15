import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
from numpy import linalg as LA
#subprocess.Popen('python3 6.py', executable='/bin/bash', shell=True)
d = 10
E0 = 1.05e-34*1.05e-34*np.pi*np.pi/2/0.067/9.1e-31/d/d/1e-18/1.6e-19*1000
#d=20*(10**(-7))
#E0 = 1.05e-34*1.05e-34*np.pi*np.pi/2/9.1e-31/d/d/1e-18/1.6e-19*1000
N_U_Steps = int(input('введите число ступеней потенциальной энергии: ')) #число ступенек потенциальной энергии, =1 в случае классической задачи
step = [0]
energy = [0]
coordinate = [0]
for i in range(N_U_Steps):
    step.append(i+1)

    coordinate.append(0.5)
for i in range(N_U_Steps):
    if i % 2 == 0:
        energy.append(100/E0)
    else:
        energy.append(0)
#print(stupenka,energy,coordinate)
step.append(2)
energy.append(0)
coordinate.append(1)


a = np.array([[step],[energy],[coordinate]])
#print(a)
#print(a)
a1 = np.reshape(a,a.size,order='F')
U_steps = np.reshape(a1,(3,-1))
print(U_steps)
#pi_const = cm.sqrt(2*0.067*9.10938188e-28)/1.054571596e-27
def Bloch_cond(momentum,Energy):
    mass_of_DT1 = []
    mass_of_DT2 = []
    for i in range(N_U_Steps):
        DT1 = np.array([[1,0],
                        [0,1]])
        mass_of_DT1.append(DT1)
        DT2 = np.array([[1, 0],
                        [0, 1]])
        mass_of_DT2.append(DT2)
    T1 = np.array([
            [1,0],
            [0,1]
        ],dtype='complex')
    for i in range(N_U_Steps):
        ki = cm.pi*cm.sqrt(Energy - U_steps[i + 1, 1])
        #print(ki)
        if abs(ki) < 1e-100: #тут цикл чтоб убрать деление на 0 разложила экспоненту и пришла к замечательному пределу
            TI = np.array([[cm.cos(ki * (U_steps[i + 2, 2] - U_steps[i + 1, 2])),
                  U_steps[i + 2, 2] - U_steps[i + 1, 2]],
            [-ki * cm.sin(ki * (U_steps[i + 2, 2] - U_steps[i + 1, 2])),
                  cm.cos(ki * (U_steps[i + 2, 2] - U_steps[i + 1, 2]))]],dtype='complex')
            #print(TI)
            #print(TI.size)
        else:
            TI = np.array([[cm.cos(ki * (U_steps[i + 2, 2] - U_steps[i + 1, 2])),
                  1 / ki * cm.sin(ki * (U_steps[i + 2, 2] - U_steps[i + 1, 2]))],
                  [-ki * cm.sin(ki * (U_steps[i + 2, 2] - U_steps[i + 1, 2])),
                  cm.cos(ki * (U_steps[i + 2, 2] - U_steps[i + 1, 2]))]],dtype='complex')
        if abs(ki) < 1e-100:
            DTI = np.array([[-pow((U_steps[i+2,2]-U_steps[i+1,2]),2),-1/3*pow((U_steps[i+2,2]-U_steps[i+1,2]),3)],
                        [-(U_steps[i+2,2]-U_steps[i+1,2])-(U_steps[i+2,2]-U_steps[i+1,2])*cm.cos(ki*(U_steps[i+2,2]-U_steps[i+1,2])),-pow((U_steps[i+2,2]-U_steps[i+1,2]),2)]],dtype='complex')
        else:
            DTI = np.array([[-1/ki*(U_steps[i+2,2]-U_steps[i+1,2])*cm.sin(ki*(U_steps[i+2,2]-U_steps[i+1,2])),-1/ki/ki/ki*cm.sin(ki*(U_steps[i+2,2]-U_steps[i+1,2]))+1/ki/ki*(U_steps[i+2,2]-U_steps[i+1,2])*cm.cos(ki*(U_steps[i+2,2]-U_steps[i+1,2]))],
                        [-1/ki*cm.sin(ki*(U_steps[i+2,2]-U_steps[i+1,2]))-(U_steps[i+2,2]-U_steps[i+1,2])*cm.cos(ki*(U_steps[i+2,2]-U_steps[i+1,2])),-1/ki*(U_steps[i+2,2]-U_steps[i+1,2])*cm.sin(ki*(U_steps[i+2,2]-U_steps[i+1,2]))]])
        for j in range(N_U_Steps):
            T5 = mass_of_DT1[j]
            if j==i:
                T6 = DTI.dot(T5)
            else:
                T6 = TI.dot(T5)
            mass_of_DT2[j] = T6
        for j in range(N_U_Steps):
            mass_of_DT1[j] = mass_of_DT2[j]
        T2 = TI.dot(T1)
        T1 = T2
    k0 = cm.pi*cm.sqrt(Energy)
    N0_obratnaya = np.array([[0.5, 0.5/(1j*k0)],
                             [0.5, -0.5/(1j*k0)]],dtype='complex')
    for i in range(N_U_Steps):
        T5=mass_of_DT1[i]
        T6 = N0_obratnaya.dot(T5)
        mass_of_DT2[i] = T6
    for i in range(N_U_Steps):
        mass_of_DT1[i] = mass_of_DT2[i]
    T2 = N0_obratnaya.dot(T1)
    deriv_N0_obratnaya = np.array([[0,-1/2/1j/k0/k0/k0],
                                   [0,1/2/1j/k0/k0/k0]],dtype='complex')
    T3 = deriv_N0_obratnaya.dot(T1)
    T1 = T2
    M0 = np.array([[cm.exp(1j*k0*U_steps[1,2]),cm.exp(-1j*k0*U_steps[1,2])],
                   [1j*k0*cm.exp(1j*k0*U_steps[1,2]),-1j*k0*cm.exp(-1j*k0*U_steps[1,2])]],dtype='complex')
    for i in range(N_U_Steps):
        T5 = mass_of_DT1[i]
        T6 = T5.dot(M0)
        mass_of_DT2[i] = T6
    for i in range(N_U_Steps):
        mass_of_DT1[i] = mass_of_DT2[i]
    T2 = T1.dot(M0)
    T4 = T3.dot(M0)
    DM0 = np.array([[1j*U_steps[1,2]/k0*cm.exp(1j*k0*U_steps[1,2]),-1j*U_steps[1,2]/k0*cm.exp(-1j*k0*U_steps[1,2])],
                    [(1j/k0-U_steps[1,2])*cm.exp(1j*k0*U_steps[1,2]),(-1j/k0-U_steps[1,2])*cm.exp(-1j*k0*U_steps[1,2])]],dtype='complex')
    T3 = T1.dot(DM0)
    DR = T3+T4
    for i in range(N_U_Steps):
        DR = DR + mass_of_DT2[i]
    detDR = DR[0, 0] * DR[1, 1] - DR[0, 1] * DR[1, 0]
    if abs(detDR) > 1e-200:
        DRM1 = np.array([[2/cm.pi/cm.pi*DR[1,1]/detDR,-2/cm.pi/cm.pi*DR[0,1]/detDR],
                         [-2/cm.pi/cm.pi*DR[1,0]/detDR,2/cm.pi/cm.pi*DR[0,0]/detDR]],dtype='complex')

    R = np.array([[T2[0,0]-cm.exp(1j*momentum),T2[0,1]],
                  [T2[1,0],T2[1,1]-cm.exp(1j*momentum)]],dtype='complex')
    A = DRM1.dot(R)
    (delta,V) = LA.eig(A)
    #print(delta)
    #delta1 = (A[0, 0] + A[1, 1] + pow((A[0, 0] + A[1, 1]) * (A[0, 0] + A[1, 1]) - 4 * (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]), 0.5)) / 2
    #delta2 = (A[0, 0] + A[1, 1] - pow((A[0, 0] + A[1, 1]) * (A[0, 0] + A[1, 1]) - 4 * (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]), 0.5)) / 2
    #print(delta)
    return delta


def Dispersion(am_disp:int, energy_max:float, energy_points:int, momentum_points:int, max_dev:float, max_iter:int):
    Disp1 = []
    Disp2 = []
    #Disp_total = []
    list_of_momentum = []
    for i in range(momentum_points):
        momentum = -cm.pi+0.01 +i*(cm.pi-0.02)/(momentum_points-1)
        list_of_momentum.append(momentum)
        k = 0
        j = 1
        while (k <= am_disp - 1) and (j < energy_points - 1):
            en1 = energy_max/(energy_points - 1) * j
            delta1 = Bloch_cond(momentum, en1)
            en2 = energy_max/(energy_points - 1) * (j + 1)
            delta2 = Bloch_cond(momentum, en2)

            if (delta1[0].real * delta2[0].real < 0) and (delta1[0].real < 0):
                l = 0
                while (abs(delta1[0].real) > max_dev) and (l < max_iter):
                    delta1 = Bloch_cond(momentum, en1)
                    en1 = en1 - delta1[0].real
                    l = l+1
                Disp1.append(momentum)
                Disp2.append(en1)
                #Disp_total.append(Disp1)
                #Disp_total.append(Disp2)
                k = k+1
            elif (delta1[1].real * delta2[1].real < 0) and (delta1[1].real < 0):
                l = 0
                while (abs(delta1[1].real) > max_dev) and (l < max_iter):
                    delta1 = Bloch_cond(momentum, en1)
                    en1 = en1 - delta1[1].real
                    l = l+1
                Disp1.append(momentum)
                Disp2.append(en1)
                #Disp_total.append(Disp1)
                #Disp_total.append(Disp2)
                k = k+1
            j = j+1


    return Disp1, Disp2

D = Dispersion(5,2000/E0,1000,100,0.001,1000)
#print(D)

mass_of_values_of_momentum_1 = D[0]
mass_of_values_of_momentum_2 = reversed(D[0])
mass_of_values_of_momentum_2_M = []
for i in mass_of_values_of_momentum_2:
    i = i*(-1)
    mass_of_values_of_momentum_2_M.append(i)
mass_of_values_of_momentum_tot = mass_of_values_of_momentum_1 + mass_of_values_of_momentum_2_M
mass_of_values_of_energy_1 = D[1]
mass_of_values_of_energy_2 = reversed(D[1])
mass_of_values_of_energy_2_M = []
for i in mass_of_values_of_energy_2:
    i = i*1
    mass_of_values_of_energy_2_M.append(i)
mass_of_values_of_energy_normal = mass_of_values_of_energy_1 + mass_of_values_of_energy_2_M
print(mass_of_values_of_energy_normal)
mass_of_values_of_energy_normal_final = []
for i in mass_of_values_of_energy_normal:
    i = i*E0
    mass_of_values_of_energy_normal_final.append(i)
#print(mass_of_values_of_energy_normal)
#fig, ax = plt.subplots(figsize)
plt.scatter(mass_of_values_of_momentum_tot,mass_of_values_of_energy_normal_final, s=2)
plt.grid(True)
plt.show()