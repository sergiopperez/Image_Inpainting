import numpy as np
from scipy import optimize

##########################
# run the 2D file
##########################



def temporal_loop(initial_phi, damage):

    conf=1 # Select the configuration of the problem
    num, n, dx, x, epsilon_1, epsilon_2, lam, dt, tmax, ntimes, pot, theta, thetac, choicemob = initial_conditions(conf, damage)

    t=np.zeros(ntimes+1) # Time vector
    phi=np.zeros([n*n,ntimes+1]) # Density matrix
    # tic = time.clock()

    epsilon = epsilon_1

    for ti in np.arange(ntimes):

        # Euler implicit

        phi[:,ti+1],infodict, ier, mesg = optimize.fsolve(lambda phi_nplus1: Euler_implicit(phi[:,ti],phi_nplus1,phi[:,0],n,dx,dt,epsilon,lam,ntimes,pot,theta,thetac,choicemob), phi[:,ti], full_output = True)


        t[ti+1]=t[ti]+dt

        print('--------------------')
        print('Time: ',t[ti])
        print(['L1 norm of the difference between the new and old state: ',np.linalg.norm(phi[:,ti+1]-phi[:,ti],1)])

        # if np.linalg.norm(phi[:,ti+1]-phi[:,ti],1) <  0.0001 :
        #     break

        if np.linalg.norm(phi[:,ti+1]-phi[:,ti],1) > 1000:
            break

        # Second step: TWO-STEP method---sharp the edges of the images while the previous step is to execute a topological reconnection of the shape with diffused edges.

        if t[ti] > 1:
            epsilon = epsilon_2

    return phi[:,ti+1]

    # save data
    # np.save('data/rho_'+str(i), rho)
    #
    # CH_image_final[i,:] = rho[:,-1]
    #
    # # plot thr figures
    #
    # figure1 = plt.figure(figsize=(10,6.5))
    # #plt.title(r'Final $\phi$ ', fontsize=36)
    # plt.imshow(np.reshape(rho[:,0],(n,n)), cmap='Greys', vmin=-1, vmax=1,aspect='auto',extent=[x[0],x[-1],x[0],x[-1]])
    # #plt.ylabel(r'$y$', fontsize=25)
    # plt.xlabel(r'$x$', fontsize=25)
    # plt.colorbar()
    # plt.show()
    # plt.close()
    # figure1.savefig('figures/INITIAL_'+str(i)+'.png', bbox_inches='tight')
    #
    # figure2 = plt.figure(figsize=(10,6.5))
    # #plt.title(r'Final $\phi$ ', fontsize=36)
    # plt.imshow(np.reshape(rho[:,-1],(n,n)), cmap='Greys', vmin=-1, vmax=1,aspect='auto',extent=[x[0],x[-1],x[0],x[-1]])
    # #plt.ylabel(r'$y$', fontsize=25)
    # plt.xlabel(r'$x$', fontsize=25)
    # plt.colorbar()
    # plt.show()
    # plt.close()
    # figure2.savefig('figures/Inpaintings_'+str(i)+'.png', bbox_inches='tight')


########################################
# DEFINE FUNCTION: INITIAL CONDITIONS
#########################################

def initial_conditions(choice, damage):

    if choice==1: #Random initial configuration
        num=100
        n=28 # Number of cells per row
        dx=1
        x= np.linspace(-n*dx/2,n*dx/2,n)
        epsilon_1 = 1.5 # Parameter epsilon_1
        epsilon_2 = 0.5 # Parameter epsilon_2

        lam=np.full((n*n), 9000)
        lam[damage]=0
        lam = np.reshape(lam,(n,n))

        dt=0.1 # Time step
        tmax=6
        ntimes=int(tmax/dt)# 4000 # Number of time steps
        pot=1 # Choice of the potential: 1 is double-well, 2 is logarithmic
        theta=0.2 # Absolut temperature for the logarithmic potential
        thetac=1. # Critical temperature for the logarithmic potential
        choicemob=1 # Choice of mobility. 1 is constant mobility, 2 is 1-rho**2


    return num, n, dx, x, epsilon_1, epsilon_2, lam, dt, tmax, ntimes, pot, theta, thetac, choicemob

########################
# DEFINE FUNCTION: FLUX
#######################


def Euler_implicit(phi_n,phi_nplus1,phi_0,n,dx,dt,epsilon,lam,ntimes,pot,theta,thetac,choicemob):

    # Create matrix phi_n

    phi_n=np.reshape(phi_n,(n,n))
    phi_nplus1=np.reshape(phi_nplus1,(n,n)) # phi_n in the future phi_n.shape = (28,28)

    # Define variation of free energy

    # a) Hc: contractive (convex) part of the free energy (treated implicitly))
    # Hc1 is the first derivative of Hc

    Hc1= Hc1_con(phi_nplus1,pot,theta)

    # b) He: expansive (concave) part of the free energy (treated explicitly)
    # He1 is the first derivative of He

    He1= He1_exp(phi_n,pot,thetac)

    # c) Laplacian (treated semi-implicitly)
    #see the difference for cells in the middle, at the corner, lying on the line

    Lap=np.zeros((n,n))

    Lap[1:-1,1:-1]=epsilon**2*(-4*phi_n[1:-1,1:-1]+phi_n[0:-2,1:-1]+phi_n[2:,1:-1]+phi_n[1:-1,0:-2]+phi_n[1:-1,2:]\
    -4*phi_nplus1[1:-1,1:-1]+phi_nplus1[0:-2,1:-1]+phi_nplus1[2:,1:-1]+phi_nplus1[1:-1,0:-2]+phi_nplus1[1:-1,2:])/dx**2/2.#cells in the middle

    Lap[1:-1,0]=epsilon**2*(-3*phi_n[1:-1,0]+phi_n[0:-2,0]+phi_n[2:,0]+phi_n[1:-1,1]\
    -3*phi_nplus1[1:-1,0]+phi_nplus1[0:-2,0]+phi_nplus1[2:,0]+phi_nplus1[1:-1,1])/dx**2/2.#cells on the left line

    Lap[1:-1,-1]=epsilon**2*(-3*phi_n[1:-1,-1]+phi_n[0:-2,-1]+phi_n[2:,-1]+phi_n[1:-1,-2]\
    -3*phi_nplus1[1:-1,-1]+phi_nplus1[0:-2,-1]+phi_nplus1[2:,-1]+phi_nplus1[1:-1,-2])/dx**2/2.#cells on the right line

    Lap[0,1:-1]=epsilon**2*(-3*phi_n[0,1:-1]+phi_n[0,0:-2]+phi_n[0,2:]+phi_n[1,1:-1]\
    -3*phi_nplus1[0,1:-1]+phi_nplus1[0,0:-2]+phi_nplus1[0,2:]+phi_nplus1[1,1:-1])/dx**2/2.#cells on the top line

    Lap[-1,1:-1]=epsilon**2*(-3*phi_n[-1,1:-1]+phi_n[-1,0:-2]+phi_n[-1,2:]+phi_n[-2,1:-1]\
    -3*phi_nplus1[-1,1:-1]+phi_nplus1[-1,0:-2]+phi_nplus1[-1,2:]+phi_nplus1[-2,1:-1])/dx**2/2.#cells on the bottom line

    Lap[0,0]=epsilon**2*(-2*phi_n[0,0]+phi_n[1,0]+phi_n[0,1]\
    -2*phi_nplus1[0,0]+phi_nplus1[1,0]+phi_nplus1[0,1])/dx**2/2.#cells on the top left corner

    Lap[0,-1]=epsilon**2*(-2*phi_n[0,-1]+phi_n[1,-1]+phi_n[0,-2]\
    -2*phi_nplus1[0,-1]+phi_nplus1[1,-1]+phi_nplus1[0,-2])/dx**2/2.#cells on the top right corner

    Lap[-1,0]=epsilon**2*(-2*phi_n[-1,0]+phi_n[-2,0]+phi_n[-1,1]\
    -2*phi_nplus1[-1,0]+phi_nplus1[-2,0]+phi_nplus1[-1,1])/dx**2/2.#cells on the bottom left corner

    Lap[-1,-1]=epsilon**2*(-2*phi_n[-1,-1]+phi_n[-2,-1]+phi_n[-1,-2]\
    -2*phi_nplus1[-1,-1]+phi_nplus1[-2,-1]+phi_nplus1[-1,-2])/dx**2/2.#cells on the bottom right corner

    # Compute (n-1) u velocities

    uhalf=-(Hc1[1:,:]-He1[1:,:]-Lap[1:,:]-Hc1[0:-1,:]+He1[0:-1,:]+Lap[0:-1,:])/dx

    # Upwind u velocities

    uhalfplus=np.zeros((n-1,n))
    uhalfminus=np.zeros((n-1,n))
    uhalfplus[uhalf > 0]=uhalf[uhalf > 0]
    uhalfminus[uhalf < 0]=uhalf[uhalf < 0]

    # Compute (n-1) v velocities

    vhalf=-(Hc1[:,1:]-He1[:,1:]-Lap[:,1:]-Hc1[:,0:-1]+He1[:,0:-1]+Lap[:,0:-1])/dx


    # Upwind u velocities


    vhalfplus=np.zeros((n,n-1))
    vhalfminus=np.zeros((n,n-1))
    vhalfplus[vhalf > 0]=vhalf[vhalf > 0]
    vhalfminus[vhalf < 0]=vhalf[vhalf < 0]


    # Compute (n+1,n) x fluxes, including no-flux boundary conditions

    Fxhalf=np.zeros((n+1,n))

    # 1st order
    Fxhalf[1:-1,:]=uhalfplus*mobility(phi_n[0:-1,:],choicemob)+uhalfminus*mobility(phi_n[1:,:],choicemob)

    # Compute (n+1) y fluxes, including no-flux boundary conditions

    Fyhalf=np.zeros((n,n+1))

    # 1st order
    Fyhalf[:,1:-1]=vhalfplus*mobility(phi_n[:,0:-1],choicemob)+vhalfminus*mobility(phi_n[:,1:],choicemob)


    # initial state

    phi_0 = np.reshape(phi_0, (np.shape(phi_nplus1)))


    #b)define lambda

    E_i=phi_nplus1-phi_n+(Fxhalf[1:,:]-Fxhalf[:-1,:]+Fyhalf[:,1:]-Fyhalf[:,:-1])*dt/dx + lam*(phi_0-phi_nplus1)*dt

    return np.reshape(E_i,n*n) #E_i.shape = (784,)



#################################################
# DEFINE FUNCTION: H CONTRACTIVE DERIVATIVE
##################################################
def Hc1_con(rho,pot,theta):
    if pot==1:# Double well
        Hc1=rho**3
    elif pot==2 and theta!=0:# Logarithmic
        Hc1=theta/2.*np.log(np.divide(1+rho,1-rho))
        greater1_index = np.append(rho,0) > 1
        lowerminus1_index = np.append(rho,0) < -1
        Hc1[greater1_index[:-1]]=9999999999999999999999999
        Hc1[lowerminus1_index[:-1]]=-9999999999999999999999
    else:
        Hc1=np.zeros(np.shape(rho))
    return Hc1

####################################################
# DEFINE FUNCTION: H CONTRACTIVE 2 DERIVATIVE
#####################################################3
def Hc2_con(rho,pot,theta):
    if pot==1:# Double well
        Hc2=3*rho**2
    elif pot==2 and theta!=0:# Logarithmic
        if np.abs(rho)<=1:
            Hc2=theta/(1-rho**2)
        else:
            Hc2=np.inf
    else:
        Hc2=np.zeros(np.shape(rho))
    return Hc2

###############################################
# DEFINE FUNCTION: H EXPANSIVE 2 DERIVATIVE
###############################################
def He1_exp(rho,pot,thetac):
    if pot==1:# Double well
        He1=rho
    elif pot==2 and thetac!=0:# Logarithmic
        He1=thetac*rho
    else:
        He1=np.zeros(np.shape(rho))
    return He1

##############################
# DEFINE FUNCTION: MOBILITY
##################################
def mobility(rho,choicemob):
    if choicemob==1:
        m=np.zeros(np.shape(rho))
        m[:]=1
    elif choicemob==2:
        m=1-rho**2
    return m
