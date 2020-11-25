import numpy as np
import numpy.random as rn
import numpy.linalg as la
import matplotlib.pyplot as plt


def frank_wolfe(x, A, b, t, gam):
    # update x (your code here)
    grad= np.matmul(A.transpose(),np.dot(A,x)-b)
    idx_s = np.argmax(np.abs(grad))
    s = np.zeros(x.size)
    s[idx_s]  = gam *np.sign(-grad[idx_s])
    x += 2.0/(t+2.0)*(s-x)
    return x


def subgradient(x, A, b, t, lam, c=1e-5):
    # update x (your code here), set c above
    grad = np.matmul(A.transpose(),np.dot(A,x)-b) +lam*np.sign(x)
    x = x - c/np.sqrt(t+1)*grad
    return x

# add BTLS variants and include them in main/descent below
def objective(x, A, b):
    return 0.5*np.dot(np.dot(A,x)-b, np.dot(A,x)-b)

def frank_wolfe_BTLS(x, A, b, t, gam, alpha=1, c=0.5, rho=0.9):
    grad= np.matmul(A.transpose(),np.dot(A,x)-b)
    idx_s = np.argmax(np.abs(grad))
    s = np.zeros(x.size)
    s[idx_s]  = gam *np.sign(-grad[idx_s])
    p=s-x
    while objective(x+alpha*p, A, b)>objective(x, A, b)+c*alpha*np.dot(grad,p):
        alpha=alpha*rho
    # print("alpha={}".format(alpha))
    x += alpha*p
    return x    

def objective_reg(x, A, b, lam):
    return 0.5*np.dot(np.dot(A,x)-b, np.dot(A,x)-b)+lam*la.norm(x,1) 

def subgradient_BTLS(x, A, b, t, lam, alpha=1, c=0.5, rho=0.9):
    # update x (your code here), set c above
    grad = np.matmul(A.transpose(),np.dot(A,x)-b) +lam*np.sign(x)
    while objective_reg(x-alpha*grad, A, b, lam)>objective_reg(x, A, b, lam)-c*alpha*np.dot(grad,grad):
        alpha=alpha*rho
    # print("alpha={}".format(alpha))
    x = x - alpha*grad
    return x

def descent(update, A, b, reg, T=int(1e4)):
    x = np.zeros(A.shape[1])
    error = []
    l1 = []
    for t in range(T):
        # update A (either subgradient or frank-wolfe)
        x = update(x, A, b, t, reg)
        #x =subgradient(x, A, b, t, reg, c=1e-5)
        # record error and l1 norm
        if (t % 1 == 0) or (t == T - 1):
            error.append(la.norm(np.dot(A, x) - b))
            l1.append(np.sum(abs(x)))

            assert not np.isnan(error[-1])

    return x, error, l1


def main(T=int(1e3)):
    A = np.load("A.npy")
    b = np.load("b.npy")

    # modify regularization parameters below
    x_sg, error_sg, l1_sg = descent(subgradient, A, b, reg=1.2, T=T)
    x_fw, error_fw, l1_fw = descent(frank_wolfe, A, b, reg=1.2, T=T)
    # add BTLS experiments

    x_sg_BTLS, error_sg_BTLS, l1_sg_BTLS = descent(subgradient_BTLS, A, b, reg=1.2, T=T)
    x_fw_BTLS, error_fw_BTLS, l1_fw_BTLS = descent(frank_wolfe_BTLS, A, b, reg=1.2, T=T)
    # add plots for BTLS
    plt.clf()
    plt.plot(error_sg, label='Subgradient')
    plt.plot(error_fw, label='Frank-Wolfe')
    plt.plot(error_sg_BTLS, label='Subgradient-BTLS')
    plt.plot(error_fw_BTLS, label='Frank-Wolfe-BTLS')
    plt.title('Error')
    plt.legend()
    plt.savefig('error.eps')

    plt.clf()
    plt.plot(l1_sg, label='Subgradient')
    plt.plot(l1_fw, label='Frank-Wolfe')
    plt.plot(l1_sg_BTLS, label='Subgradient-BTLS')
    plt.plot(l1_fw_BTLS, label='Frank-Wolfe-BTLS')
    plt.title("$\ell^1$ Norm")
    plt.legend()
    plt.savefig('l1.eps')

    print("last error_sg={}".format(error_sg[-1]))
    print("last error_fw={}".format(error_fw[-1]))
    print("last error_fw_BTLS={}".format(error_fw_BTLS[-1]))
    print("last error_sg_BTLS={}".format(error_sg_BTLS[-1]))
    print("last l1_sg={}".format(l1_sg[-1]))
    print("last l1_fw={}".format(l1_fw[-1]))
    print("last l1_sg_BTLS={}".format(l1_sg_BTLS[-1]))
    print("last l1_fw_BTLS={}".format(l1_fw_BTLS[-1]))
if __name__ == "__main__":
    main()
