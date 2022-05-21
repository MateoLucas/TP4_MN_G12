import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def ruku4(f, t0, tf, delta_t, x0):
    assert delta_t > 0, 'Error: se requiere delta_t mayor a 0'
    assert tf > t0, 'Error: el intervalo [t0, tf] está mal definido'
    
    xk = x0
    tk = t0

    t_array = np.arange(t0, tf, delta_t)
    x_array = x0

    for i in t_array:
        if i == 0: continue #omito rellenar el primer valor, pues es x0
        f1 = f(tk, xk)
        f2 = f(tk + (delta_t/2), xk + ((delta_t*f1)/2))
        f3 = f(tk + (delta_t/2), xk + ((delta_t*f2)/2))
        f4 = f(tk+delta_t , xk+delta_t*f3)

        xk = xk + (f1+2*f2+2*f3+f4)*delta_t/6
        x_array = np.vstack([x_array, xk])

    return t_array, x_array


def  hodgkinhuxley():
    def alpha_n(v): return (0.01*((v+55)/(1-(np.e**(-1*((v+55)/(10)))))))
    def alpha_m(v): return (0.01*((v+40)/(1-(np.e**(-1*((v+40)/(10)))))))
    def alpha_h(v): return 0.07*(np.e**(-1*((v+65)/20)))
    def beta_n(v): return 0.125*(np.e**(-1*((v+65)/80)))
    def beta_m(v): return 4*(np.e**(-1*((v+65)/18)))
    def beta_h(v): return (1/(1+(np.e**(-1*((v+35)/10)))))
    gNa = 120
    gK = 36
    gL = 0.3
    vNa = 50
    vK=-77
    vL=-54.4
    C=1





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


