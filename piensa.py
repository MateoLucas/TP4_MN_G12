import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#RK4 METHOD


def ruku4(f,t0,tf,h,x0):
    ite = round((tf-t0)/h)+1  # Se calcula la cantidad de veces a iterar
    dim = np.size(x0) # Dimensión del problema
    t = np.linspace(t0,tf,ite)
    x = np.empty((np.size(t),dim))
    x[0,:] = np.transpose(x0)
    for i in range(0,ite-1):
        f1=f(t[i],x[i,:])
        f2=f(t[i]+(h/2),x[i,:]+(h*f1/2))
        f3=f(t[i]+(h/2),x[i,:]+(h*f2/2))
        f4=f(t[i]+h,x[i,:]+h*f3)
        
        x[i+1,:] = x[i,:]+((f1+2*f2+2*f3+f4)*h/6.0)

    return t,x

#HH
#definimos fuinciones y constantes afuera de la funcion para no
#redefinirlas cada vez que hacemos el llamado
#Pasar incognitas como [v,n,m,h]
def alpha_n(v): return (0.01*((v+55)/(1-(np.e**(-1*((v+55)/(10)))))))
def alpha_m(v): return (0.01*((v+40)/(1-(np.e**(-1*((v+40)/(10)))))))
def alpha_h(v): return 0.07*(np.e**(-1*((v+65)/20)))
def beta_n(v): return 0.125*(np.e**(-1*((v+65)/80)))
def beta_m(v): return 4*(np.e**(-1*((v+65)/18)))
def beta_h(v): return (1/(1+(np.e**(-1*((v+35)/10)))))
gNa = 120.0
gK = 36.0
gL = 0.3
vNa = 50.0
vK=-77.0
vL=-54.4
i0 = 2
C=1

def HH_eq(t,X):
    dv = (1/C)*(i0-gNa*(X[2]**3)*X[3]*(X[0]-vNa)-gK*(X[1]**4)*(X[0]-vK)-gL*(X[0]-vL))
    dn = alpha_n(X[0])*(1-X[1])-beta_n(X[0])*X[1]
    dm = alpha_m(X[0])*(1-X[2])-beta_m(X[0])*X[2]
    dh = alpha_h(X[0])*(1-X[3])-beta_h(X[3])*X[3]
    return np.array([dv,dn,dm,dh])

def  hodgkinhuxley():

    x0 = np.array([-65, 0,0,0])
    t,x =ruku4(HH_eq,0,200,1e-4,x0)

    #gráfico
    HH = plt.figure("Potencial de Acción HH")
    plt.xlabel("t[ms]")
    plt.ylabel("V(t)[mV]")
    plt.plot(t,x[:,0],color = "#40eb34")


    plt.show()
    return t,x





a=hodgkinhuxley()







def test():
    t0, tf, delta_t = 0, 150, 20e-4

    test1 =[[lambda t, x: np.array([[0, -1], [1, 0]], dtype=float) @ x, 
            np.array([1,1], dtype=float), 
            lambda t: np.array([np.cos(t)-np.sin(t), np.sin(t)+np.cos(t)], dtype=float)],
            [lambda t, x: np.array([[-1, 0, 0], [0, -3, 0], [0, 0, -4]], dtype=float) @ x, 
            np.array([15, 6, 9], dtype=float), 
            lambda t: np.array([15*np.exp(-t), 6*np.exp(-3*t), 9*np.exp(-4*t)], dtype=float)],
            [lambda t, x: np.array([[3, -18], [2, -9]], dtype=float) @ x, 
            np.array([14, 5], dtype=float), 
            lambda t: np.array([15*np.exp(-3*t)-(6*t+1)*np.exp(-3*t), 5*np.exp(-3*t)-2*t*np.exp(-3*t)], dtype=float)],
            [lambda t, x: np.array([[-1, -2, -2, 6], [1, 1, 3, -4], [0, 0, 1, -2], [0, 0, 1, -1]], dtype=float) @ x,
            np.array([-2, 1, 2, 1], dtype=float),
            lambda t: np.array([-2*np.cos(t)+2*np.sin(t), np.cos(t)+np.sin(t), 2*np.cos(t), np.cos(t)+np.sin(t)], dtype=float)],
            [lambda t, x: np.array([[-2, -1], [-1, -2]], dtype=float) @ x, 
            np.array([0, 2], dtype=float), 
            lambda t: np.array([np.exp(-3*t)-np.exp(-t), np.exp(-3*t)+np.exp(-t)], dtype=float)]]
    
    n=1

    for i in test1:
        t, x = ruku4(i[0], t0, tf, delta_t, i[1])
        y = np.array([i[2](a) for a in t])
        print('Test '+str(n)+', error obtenido: '+str(np.linalg.norm(y-x)))
        n+=1

    test2 = [[lambda t, x: np.array([x[0]*(1-x[0])- x[0]*x[1], 2*x[1]*(1-0.5*x[1]**2)-3*x[0]**2*x[1]], dtype=float),
            np.array([0.1, 0.1], dtype=float)],
            [lambda t, x: np.array([-2*t*x[0], -5*x[1]**2], dtype=float),
            np.array([0.01, 0.01], dtype=float)],
            [lambda t, x: np.array([np.sin(x[0])+np.cos(x[1]), np.sin(x[0])-np.cos(x[1])], dtype=float),
            np.array([-1, -2], dtype=float)],
            [lambda t, x: np.array([-np.exp(-x[1])/x[0], np.cos(x[0])/x[1]]),
            np.array([1, 3], dtype=float)],
            [lambda t, x: np.array([np.exp(-x[1]**2-x[0]), np.sin(np.cos(x[1]+x[0]))], dtype=float),
            np.array([1, 2], dtype=float)]]

    for i in test2:
        t, x = ruku4(i[0], t0, tf, delta_t, i[1])
        y = solve_ivp(i[0], [t0, tf], i[1], t_eval=t) #Se pasa el arreglo de tiempo para comparar valores en iguales puntos
        print('Test '+str(n)+', error obtenido: '+str(np.linalg.norm(np.transpose(y.y)-x)))
        n+=1


