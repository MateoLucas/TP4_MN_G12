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

def comprobarOsc(y):
    i = 0
    j = -1

    while(y[i] > y[i+1]): #busco el primer mínimo
        i += 1
    while(y[i] < y[i+1]): #voy hasta el máximo
        i += 1
    y1 = y[i] #registro el valor de y para el primer máximo
    while(y[i] > y[i+1]): #voy hasta el mínimo
        i += 1
    y2 = y[i] #registro el valor de y para el primer mínimo
    while(y[j] > y[j-1]): #repito lo mismo desde "atrás"
        j -= 1
    while(y[j] < y[j-1]):
        j -= 1
    y3 = y[j]
    while(y[j] > y[j-1]):
        j -= 1
    y4 = y[j]
    
    return (y3-y4)/(y1-y2) > 0.1

def higginsselkov(): #x = (s, p)    
    t0, tf, delta_t = 0, 600, 0.1

    x0 = np.array([2, 3])
    
    g = lambda t, u: np.array([v0-0.23*u[0]*u[1]**2, 0.23*u[0]*u[1]**2-0.4*u[1]], dtype=float);

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle('Solución a la ecuación de Higgins-Selkov para distintos valores de v0')
    
    l1 = 0.48
    l2 = 0.6

    x, xl1, xl2, t = [], [], [], []

    v0 = l1
    t, xl1 = ruku4(g, t0, tf, delta_t, x0)

    ax1.plot(t, xl1[:, 0])    
    ax1.plot(t, xl1[:, 1])
    ax1.set(ylabel='v0 = '+str(l1))
    ax1.legend(['[F6P]', '[ADP]'])
    ax1.grid()

    v0 = l2
    t, xl2 = ruku4(g, t0, tf, delta_t, x0)

    ax3.plot(t, xl2[:, 0])
    ax3.plot(t, xl2[:, 1])
    ax3.set(ylabel='v0 = '+str(l2), xlabel='tiempo [s]')
    ax3.legend(['[F6P]', '[ADP]'])
    ax3.grid()

    v0 = 0.5*(l1+l2)

    MAX_ITER = 100
    i = 0

    while(i < MAX_ITER):
        t, x = ruku4(g, t0, tf, delta_t, x0)

        s = x[:, 0]
        p = x[:, 1]

        if(comprobarOsc(s) and comprobarOsc(p)):
            l1 = v0
        else:
            l2 = v0

        v0 = 0.5*(l1+l2)
        i += 1

    ax2.plot(t, x[:, 0])
    ax2.plot(t, x[:, 1])
    ax2.set(ylabel='v0 = '+str(v0))
    ax2.legend(['[F6P]', '[ADP]'])
    ax2.grid()

    plt.show()

    print('El valor de v0 para el cual s y p entran en régimen oscilatorio es aproximadamente '+str(v0))

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

higginsselkov()
test()