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
def alpha_m(v): return (0.1*((v+40)/(1-(np.e**(-1*((v+40)/(10)))))))
def alpha_h(v): return 0.07*(np.e**(-1*((v+65)/20)))
def beta_n(v): return 0.125*(np.e**(-1*((v+65)/80)))
def beta_m(v): return 4*(np.e**(-1*((v+65)/18)))
def beta_h(v): return (1/(1+(np.e**(-1*((v+35)/10)))))
gNa = 120.0 #mV
gK = 36.0 #mV
gL = 0.3 #mV
vNa = 50.0 #mV
vK=-77.0 #mV
vL=-54.4 #mV
i0 = 2 #uA/cm3
C=1

def HH_eq(t,X):
    dv = (1/C)*(i0-gNa*(X[2]**3)*X[3]*(X[0]-vNa)-gK*(X[1]**4)*(X[0]-vK)-gL*(X[0]-vL))
    dn = alpha_n(X[0])*(1-X[1])-beta_n(X[0])*X[1]
    dm = alpha_m(X[0])*(1-X[2])-beta_m(X[0])*X[2]
    dh = alpha_h(X[0])*(1-X[3])-beta_h(X[0])*X[3]
    return np.array([dv,dn,dm,dh])

def  hodgkinhuxley():

    x0 = np.array([-65, 0,0,0])
    t,x =ruku4(HH_eq,0,200,1e-2,x0)
    return t,x
"""
def test():
    
    #Se testea la funcion ruku4 con un ejemplo conocido
    T0=np.array([1000]) #Temperatura inicial del objeto en K
    t0=0
    tf=1000
    m,w=ruku4 (derivadatest,t0,tf,0.01,T0)
    T=w[0]
        #Solucion analitica conocida para comparar
    h=100 #Unidades de W/m**2K coeficiente de transferencia de energia
    A=0.00001 #Area superficial en unidades de m**2
    p=7800 #densidad del objeto en unidades de kg/cm**2
    C=500 #Unidades de J/kgK Capacidad calorifica especifica
    V=0.00000001 #Volumen en m**3
    Ta=298 #Unidades de K temperatura ambiente 
    T0=1000 #Temperatura inicial en K
    M=Ta+(T0-Ta)*np.exp(-((h*A)/(p*C*V))*m) #solucion analitica
    plot_grafico(m,[T,M],"Tiempo [s]","Temperatura [K]",refs=["Solucion con ruku4","Solucion analitica"])
    plot_grafico(m,[np.abs(T-M)],"Tiempo [s]","Error [K]",refs=0)
    
    #Se testea HH utilizando una funcion de la libreria de Python
    t0=0
    tf=200
    x0=np.array([-65, 0, 0, 0])
    t,x=ruku4(derivada, t0 ,tf ,0.01,x0)
    v=x[0]
    s45=solve_ivp(derivada,[0,200],x0,method='RK45', t_eval=t, rtol=1e-13, atol=1e-14)
    x45=s45.y
    plot_grafico(t,[x45[0,:]],"Tiempo [ms]","Potencial de membrana [mV]",refs=0)
    error1=np.abs(x45[0,:]-v)
    plot_grafico(t,[error1],"Tiempo [ms]","Potencial de membrana [mV]",refs=0)
    
    #Justificacion de la eleccion del paso 
    t2,x2=ruku4(derivada, t0, tf,0.01/2,x0) #Se utiliza la mitad del paso elegido 
    v2=x2[0]
    s45bis=solve_ivp(derivada,[0,200],x0,method='RK45', t_eval=t2, rtol=1e-13, atol=1e-14)
    x45bis=s45bis.y
    error2 = np.abs (x45bis[0,:]-v2) #error del orden de 1e-6 con 0.01 y del orden de 1e-9 con 0.001
    plot_grafico(t2,[error2],"Tiempo [ms]","Potencial de membrana [mV]",refs=0)
    print(max(error1)/max(error2)) #Error relativo da 15.9934107 con 0.01 y 13.228723 con 0.001 
    
    return 

"""
def test():
    #Sistema de HH
    x0 = np.array([-65, 0,0,0])
    t,x =ruku4(HH_eq,0,200,1e-2,x0)

    HH = plt.figure("Potencial de Acción HH")
    HH.add_subplot(2,1,1)
    plt.xlabel("t[ms]")
    plt.ylabel("Probabilidad de compuertas abiertas")
    plt.plot(t,x[:,1],color = 'r', label='n(t)')
    plt.plot(t,x[:,2],color = 'g', label='m(t)')
    plt.plot(t,x[:,3],color = 'b', label='h(t)')
    plt.legend()
    HH.add_subplot(2,1,2)
    plt.xlabel("t[ms]")
    plt.ylabel("V(t)[mV]")
    plt.plot(t,x[:,0],color = "#fcbe03")
    plt.show()

    #Test 2

    y0 = np.array([1,1])
    t0 = -5
    tf = np.pi
    def dy(t,y): return np.array([y[0],-200*np.sin(t)])
    t,y = ruku4(dy,t0,tf,0.1,y0)
    test2 = plt.figure("Prueba de 2 funciones básicas")
    plt.xlabel("t")
    plt.plot(t,y[:,0],color = "#40eb34",label= "y'=y")
    plt.plot(t,y[:,1],color = "#0f03fc",label= "z' = -200sin(t)")
    plt.legend()
    plt.show()

    #Test 3 

    y0 = np.array([0,1]) # y0,y1
    t0 = -5
    tf = 4
    def dy(t,y): return np.array([2*y[0]-3*y[1],-y[0]+4*y[1]])
    t,y = ruku4(dy,t0,tf,0.1,y0)
    test3 = plt.figure("Prueba 3")
    plt.xlabel("t")
    plt.plot(t,y[:,0],color = "#40eb34",label= "y0' = 2y0 -3y1")
    plt.plot(t,y[:,1],color = "#0f03fc",label= "y1' = -y0+4y1")
    plt.legend()
    plt.show()

    #Test 4 - Transitorio de RC paralelo REVISAR

    y0 = np.array([0,0,0]) # iR, iC, U
    t0 = 0
    tf = 0.1
    U = 5
    R = 20
    C = 0.001
    def dy(t,y): return np.array([(U/R)- y[1],C*y[2],y[1]/C])
    t,y = ruku4(dy,t0,tf,0.0001,y0)
    test4 = plt.figure("Prueba con circuito RL")
    plt.xlabel("t")
    plt.plot(t,y[:,0],color = "#40eb34",label= "i' = (-U0+iR)/L")
    plt.plot(t,y[:,1],color = "#0f03fc",label= "UR'= (i')R")
    plt.plot(t,y[:,2],color = "#0f03fc",label= "UR'= (i')R")
 
    plt.legend()
    plt.show()
    


test()

