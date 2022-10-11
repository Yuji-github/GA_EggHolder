import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

def f(x1,x2):
    a=jnp.sqrt(jnp.fabs(x2+x1/2+47))
    b=jnp.sqrt(jnp.fabs(x1-(x2+47)))
    c=-(x2+47)*jnp.sin(a)-x1*jnp.sin(b)
    return c
x1=jnp.linspace(-512,512,100)
x2=jnp.linspace(-512,512,100)
X1,X2=jnp.meshgrid(x1,x2)

def plotter():
    fig=plt.figure(figsize=[12,8])
    ax=plt.axes(projection='3d')
    ax.plot_surface(X1,X2,f(X1,X2),color='red',alpha=0.7)
    ax.plot_wireframe(X1,X2,f(X1,X2),ccount=2,rcount=2, color='orange',alpha=0.8)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')
    plt.show()

plotter()

# if __name__ == '__main__':
#     create_egg_holder()
