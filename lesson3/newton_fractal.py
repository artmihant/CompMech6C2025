import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# def F(z):
#     return (z-1)*(z-2)*(z-3)

# def DF(z):
#     return 3*z**2 - 12*z + 11

def F(z):
    return z**3 - 2*z+2

def DF(z):
    return 3*z**2 - 2


# def F(z):
#     return z**3 - 1

# def DF(z):
#     return 3*z**2


def newton_step(z):
    return z - F(z)/DF(z)

X_min, X_max = -2,2
X_step = 1000

Y_min, Y_max = -2,2
Y_step = 1000

NewtonSteps = 60

XSpace = np.linspace(X_min, X_max, X_step).reshape((1,-1))
YSpace = np.linspace(Y_min, Y_max, Y_step).reshape((-1,1))


ZSpaceTjajectory = np.zeros((NewtonSteps+1,X_step,Y_step), dtype='complex')
ZSpaceTjajectory[0] = (XSpace + 1j*YSpace).reshape((X_step, Y_step))

for i in range(NewtonSteps):
    ZSpaceTjajectory[i+1] = newton_step(ZSpaceTjajectory[i])


def show(ZSpaceTjajectory):

    def to_pixel(z):
        return np.sign(z.imag)*np.arccos(z.real/abs(z))
        # return abs(z)

    fig, ax = plt.subplots()

    Image = to_pixel(ZSpaceTjajectory[0])

    im = ax.imshow(Image, cmap='hsv', extent=(X_min, X_max, Y_min, Y_max))

    def loop_animation(i):
        Image = to_pixel(ZSpaceTjajectory[i])
        im.set_array(Image)

        return im

    FPS = 1

    ani = animation.FuncAnimation(
        fig=fig, 
        func=loop_animation, 
        frames=100, 
        interval=1000/FPS,
        repeat=False
    )
    plt.show()

show(ZSpaceTjajectory)

# fig, ax = plt.subplots()


# Image = to_pixel(ZSpaceTjajectory[NewtonSteps])


# im = ax.imshow(Image, cmap='twilight_shifted', extent=(X_min, X_max, Y_min, Y_max))

# plt.show()

