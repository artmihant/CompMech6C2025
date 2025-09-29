import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# def F(z):
#     return z**3 - 1

# def dF(z):
#     return 3*z**2


def F(x):
    return x**3 - 2*x +2

def dF(x):
    return 3*x**2 - 2


# def F(z):
#     return (z-1)*(z-2)*(z-3)

# def dF(z):
#     return 3*z**2 - 12*z + 11

def G(z):
    return z - F(z)/dF(z)

X_min, X_max = -2, 2
Y_min, Y_max = -2j, 2j

X_Counts = 1000
Y_Counts = 1000

XSpace_0 = np.linspace(X_min, X_max, X_Counts).reshape(1,-1)
YSpace_0 = np.linspace(Y_min, Y_max, Y_Counts).reshape(-1,1)

ZSpace_0 = XSpace_0 + YSpace_0

Steps = 100

ZSpaceTrajectory = np.zeros((Steps+1, X_Counts, Y_Counts), dtype='complex')

ZSpaceTrajectory[0] = ZSpace_0

for i in range(Steps):
    ZSpaceTrajectory[i+1] = G(ZSpaceTrajectory[i])

# fig, ax = plt.subplots()

def to_pixel(z):
    # return np.sign(z.imag)*np.arccos(z.real/abs(z))
    return abs(z)

# fig, ax = plt.subplots()

# Image = to_pixel(F(ZSpaceTrajectory[0]))

# im = ax.imshow(Image, cmap='hsv', vmin=0, vmax=1)

# plt.show()

def show(ZSpaceTrajectory):

    def to_pixel(z):
        return np.sign(z.imag)*np.arccos(z.real/abs(z))
        # return abs(z)

    fig, ax = plt.subplots()

    Image = to_pixel(ZSpaceTrajectory[0])

    im = ax.imshow(Image, cmap='hsv')

    def loop_animation(i):
        Image = to_pixel(ZSpaceTrajectory[i])
        im.set_array(Image)

        return im

    FPS = 10

    ani = animation.FuncAnimation(
        fig=fig, 
        func=loop_animation, 
        frames=100, 
        interval=1000/FPS,
        repeat=False
    )
    plt.show()

show(ZSpaceTrajectory)





# im = ax.plot(XSpace, F(XSpace))
# ax.grid()
# plt.show()

