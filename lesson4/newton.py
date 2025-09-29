import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# def F(x):
#     return (x-1)*(x-2)*(x-3)

# def DF(x):
#     return 3*x**2 - 12*x + 11

def F(x):
    return x**3-2*x+2

def DF(x):
    return 3*x**2 - 2

fig, ax = plt.subplots()


def newton_step(x):
    return x - F(x)/DF(x)

X_min, X_max = -2.5,2.5
X_step = 1000

NewtonSteps = 100

XSpace_0 = np.linspace(X_min, X_max, X_step)

XSpaceTjajectory = np.zeros((NewtonSteps+1,X_step))

# im = ax.plot(XSpace_0, F(XSpace_0))
# ax.grid()

# plt.show()


XSpaceTjajectory[0] = XSpace_0

for i in range(NewtonSteps):
    XSpaceTjajectory[i+1] = newton_step(XSpaceTjajectory[i])


def show(ZSpaceTjajectory):

    def to_pixel(z):
        return np.sign(z.imag)*np.arccos(z.real/abs(z))
        # return abs(z)

    im = ax.plot(XSpaceTjajectory[0], XSpaceTjajectory[0],'.')

    def loop_animation(i):
        # Image = to_pixel(ZSpaceTjajectory[i])
        im[0].set_ydata(XSpaceTjajectory[i])

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

show(XSpaceTjajectory)


fig, ax = plt.subplots()


def to_pixel(x):
    return abs(x)

Image = to_pixel(XSpaceTjajectory)

im = ax.imshow(Image, cmap='hsv', vmin=0, vmax=4, aspect=0.5, extent=(X_min, X_max, NewtonSteps,0))

plt.show()




Image = to_pixel(ZSpaceTjajectory[NewtonSteps])


im = ax.imshow(Image, cmap='twilight_shifted', extent=(X_min, X_max, Y_min, Y_max))

plt.show()

