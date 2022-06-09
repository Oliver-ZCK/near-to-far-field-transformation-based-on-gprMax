import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def generate_index_array(xmin, ymin, zmin, xmax, ymax, zmax):
    index_array = np.ones((xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1), dtype=int)
    for z in range(2, zmax - zmin - 1):
        for y in range(2, ymax - ymin - 1):
            for x in range(2, xmax - xmin - 1):
                index_array[x, y, z] = False

    du = 0.002
    index = 1
    for z in range(zmax - zmin + 1):
        for y in range(ymax - ymin + 1):
            for x in range(xmax - xmin + 1):
                if index_array[x, y, z]:
                    # print("#rx: {} {} {}\n".format(round(du * (x + xmin), 3), round(du * (y + ymin), 3), round(du * (z + zmin), 3)))
                    index_array[x, y, z] = index
                    index = index + 1

    return index_array

class Box:
    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax, freq, out_path):
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax
        self.nx = self.xmax - self.xmin + 1
        self.ny = self.ymax - self.ymin + 1
        self.nz = self.zmax - self.zmin + 1
        self.freq = freq
        self.out_path = out_path
        self.f = h5py.File(self.out_path, 'r')
        self.dxdydz = self.f.attrs['dx_dy_dz']
        self.dx = self.dxdydz[0]
        self.dy = self.dxdydz[1]
        self.dz = self.dxdydz[2]
        self.dt = self.f.attrs['dt']
        self.iterations = self.f.attrs["Iterations"]
        self.box_center = (self.dx * (self.nx / 2), self.dy * (self.ny / 2), self.dz * (self.nz / 2))  # 以矩形盒左下角为原点
        c: float = 299792458.0
        mu0: float = 4e-7 * np.pi
        eps0: float = 1.0 / (mu0 * c ** 2)
        self.k = 2 * np.pi * self.freq * ((mu0 * eps0) ** 0.5)
        self.eta0: float = mu0 * c
        self.r = 1
        self.A = 1j * self.k * np.exp(-1j * self.k * self.r) / (4 * np.pi * self.r)

    def generate_index_array(self):
        self.index_array = np.ones((self.nx, self.ny, self.nz), dtype=int)
        for z in range(2, self.nz - 2):
            for y in range(2, self.ny - 2):
                for x in range(2, self.nx - 2):
                    self.index_array[x, y, z] = False

        index = 1
        for z in range(self.nz):
            for y in range(self.ny):
                for x in range(self.nx):
                    if self.index_array[x, y, z]:
                        self.index_array[x, y, z] = index
                        index = index + 1

        return self.index_array

    def read_from_out(self):
        self.bottom = np.zeros((self.nx, self.ny, 2, self.iterations, 4))
        self.top = np.zeros((self.nx, self.ny, 2, self.iterations, 4))
        self.front = np.zeros((self.nx, 2, self.nz, self.iterations, 4))
        self.back = np.zeros((self.nx, 2, self.nz, self.iterations, 4))
        self.left = np.zeros((2, self.ny, self.nz, self.iterations, 4))
        self.right = np.zeros((2, self.ny, self.nz, self.iterations, 4))

        for z in range(0, 2):
            for y in range(self.ny):
                for x in range(self.nx):
                    self.bottom[x, y, z, :, 0] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Ex'][:]
                    self.bottom[x, y, z, :, 1] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Ey'][:]
                    self.bottom[x, y, z, :, 2] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Hx'][:]
                    self.bottom[x, y, z, :, 3] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Hy'][:]

        for z in range(self.nz - 2, self.nz):
            for y in range(self.ny):
                for x in range(self.nx):
                    self.top[x, y, z - self.nz + 2, :, 0] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Ex'][:]
                    self.top[x, y, z - self.nz + 2, :, 1] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Ey'][:]
                    self.top[x, y, z - self.nz + 2, :, 2] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Hx'][:]
                    self.top[x, y, z - self.nz + 2, :, 3] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Hy'][:]

        for z in range(self.nz):
            for y in range(0, 2):
                for x in range(self.nx):
                    self.front[x, y, z, :, 0] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Ex'][:]
                    self.front[x, y, z, :, 1] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Ez'][:]
                    self.front[x, y, z, :, 2] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Hx'][:]
                    self.front[x, y, z, :, 3] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Hz'][:]

        for z in range(self.nz):
            for y in range(self.ny - 2, self.ny):
                for x in range(self.nx):
                    self.back[x, y - self.ny + 2, z, :, 0] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Ex'][:]
                    self.back[x, y - self.ny + 2, z, :, 1] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Ez'][:]
                    self.back[x, y - self.ny + 2, z, :, 2] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Hx'][:]
                    self.back[x, y - self.ny + 2, z, :, 3] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Hz'][:]

        for z in range(self.nz):
            for y in range(self.ny):
                for x in range(0, 2):
                    self.left[x, y, z, :, 0] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Ey'][:]
                    self.left[x, y, z, :, 1] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Ez'][:]
                    self.left[x, y, z, :, 2] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Hy'][:]
                    self.left[x, y, z, :, 3] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Hz'][:]

        for z in range(self.nz):
            for y in range(self.ny):
                for x in range(self.nx - 2, self.nx):
                    self.right[x - self.nx + 2, y, z, :, 0] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Ey'][:]
                    self.right[x - self.nx + 2, y, z, :, 1] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Ez'][:]
                    self.right[x - self.nx + 2, y, z, :, 2] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Hy'][:]
                    self.right[x - self.nx + 2, y, z, :, 3] = self.f['/rxs/rx' + str(self.index_array[x, y, z]) + '/' + 'Hz'][:]

        return self.bottom, self.top, self.front, self.back, self.left, self.right

    def cal_pattern(self, pattern_path):
        bottom = Bottom()
        print('bottom实例化完成')
        bottom.cal_EandH()
        print('bottom计算电场磁场完成')
        bottom.cal_JandM()
        print('bottom计算电流磁流完成')
        bottom.DFT()
        print('bottom计算频域完成')
        bottom.cal_coordinate()
        print('bottom计算坐标完成')

        top = Top()
        print('top实例化完成')
        top.cal_EandH()
        print('top计算电场磁场完成')
        top.cal_JandM()
        print('top计算电流磁流完成')
        top.DFT()
        print('top计算频域完成')
        top.cal_coordinate()
        print('top计算坐标完成')

        front = Front()
        print('front实例化完成')
        front.cal_EandH()
        print('front计算电场磁场完成')
        front.cal_JandM()
        print('front计算电流磁流完成')
        front.DFT()
        print('front计算频域完成')
        front.cal_coordinate()
        print('front计算坐标完成')

        back = Back()
        print('back实例化完成')
        back.cal_EandH()
        print('back计算电场磁场完成')
        back.cal_JandM()
        print('back计算电流磁流完成')
        back.DFT()
        print('back计算频域完成')
        back.cal_coordinate()
        print('back计算坐标完成')

        left = Left()
        print('left实例化完成')
        left.cal_EandH()
        print('left计算电场磁场完成')
        left.cal_JandM()
        print('left计算电流磁流完成')
        left.DFT()
        print('left计算频域完成')
        left.cal_coordinate()
        print('left计算坐标完成')

        right = Right()
        print('right实例化完成')
        right.cal_EandH()
        print('right计算电场磁场完成')
        right.cal_JandM()
        print('right计算电流磁流完成')
        right.DFT()
        print('right计算频域完成')
        right.cal_coordinate()
        print('right计算坐标完成')

        theta = np.arange(0, 360, 1) * np.pi / 180
        phi = np.arange(0, 360, 1) * np.pi / 180
        E_theta = np.zeros((len(theta), len(phi), 6), dtype=complex)
        E_phi = np.zeros((len(theta), len(phi), 6), dtype=complex)

        print('开始计算辐射场')
        for index1, th in enumerate(tqdm(theta)):
            for index2, ph in enumerate(phi):
                E_theta[index1, index2, 0], E_phi[index1, index2, 0] = bottom.cal_radiation(th, ph)
                E_theta[index1, index2, 1], E_phi[index1, index2, 1] = top.cal_radiation(th, ph)
                E_theta[index1, index2, 2], E_phi[index1, index2, 2] = front.cal_radiation(th, ph)
                E_theta[index1, index2, 3], E_phi[index1, index2, 3] = back.cal_radiation(th, ph)
                E_theta[index1, index2, 4], E_phi[index1, index2, 4] = left.cal_radiation(th, ph)
                E_theta[index1, index2, 5], E_phi[index1, index2, 5] = right.cal_radiation(th, ph)

        E_theta = np.sum(E_theta, axis=-1)
        E_phi = np.sum(E_phi, axis=-1)
        E_pattern = (E_theta ** 2 + E_phi ** 2) ** 0.5

        print('正在存储方向图')
        np.savez(pattern_path, theta=theta, phi=phi, E_theta=E_theta, E_phi=E_phi, E_pattern=E_pattern)

        print('完成')

class Bottom(Box):
    def __init__(self):
        super().__init__(xmin, ymin, zmin, xmax, ymax, zmax, freq, out_path)
        # 不规范,box为父类的实例化对象
        self.Ex = box.bottom[:, :, :, :, 0]
        self.Ey = box.bottom[:, :, :, :, 1]
        self.Hx = box.bottom[:, :, :, :, 2]
        self.Hy = box.bottom[:, :, :, :, 3]

    def cal_EandH(self):
        self.Ex = (self.Ex[1:-1, 1:-1, 1] + self.Ex[1:-1, 2:, 1]) / 2
        self.Ey = (self.Ey[1:-1, 1:-1, 1] + self.Ey[2:, 1:-1, 1]) / 2
        self.Hx = (self.Hx[1:-1, 1:-1, 0] + self.Hx[2:, 1:-1, 0] + self.Hx[1:-1, 1:-1, 1] + self.Hx[2:, 1:-1, 1]) / 4
        self.Hy = (self.Hy[1:-1, 1:-1, 0] + self.Hy[1:-1, 2:, 0] + self.Hy[1:-1, 1:-1, 1] + self.Hy[1:-1, 2:, 1]) / 4

    def cal_JandM(self):
        self.Jx = self.Hy
        self.Jy = - self.Hx
        self.Mx = - self.Ey
        self.My = self.Ex
        del self.Ex, self.Ey, self.Hx, self.Hy

    def DFT(self):
        self.DFT_Jx = np.zeros((self.nx - 2, self.ny - 2), dtype=complex)
        self.DFT_Jy = np.zeros((self.nx - 2, self.ny - 2), dtype=complex)
        self.DFT_Mx = np.zeros((self.nx - 2, self.ny - 2), dtype=complex)
        self.DFT_My = np.zeros((self.nx - 2, self.ny - 2), dtype=complex)
        for n in range(self.iterations):
            e = np.exp(-1j * 2 * np.pi * self.freq * (n + 1) * self.dt)
            self.DFT_Jx += self.Jx[:, :, n] * e
            self.DFT_Jy += self.Jy[:, :, n] * e
            self.DFT_Mx += self.Mx[:, :, n] * e
            self.DFT_My += self.My[:, :, n] * e
        self.DFT_Jx = self.DFT_Jx * self.dt
        self.DFT_Jy = self.DFT_Jy * self.dt
        self.DFT_Mx = self.DFT_Mx * self.dt
        self.DFT_My = self.DFT_My * self.dt
        del self.Jx, self.Jy, self.Mx, self.My

    def cal_coordinate(self):
        self.coor = np.zeros((self.nx - 2, self.ny - 2, 3))
        # 计算以矩形盒左下角为坐标原点时，各单元面中心的坐标
        for y in range(self.ny - 2):
            for x in range(self.nx - 2):
                self.coor[x, y, :] = (self.dx * (x + 1.5), self.dy * (y + 1.5), self.dz)
        # 计算以矩形盒中点为坐标原点时，各单元面中心的坐标
        self.coor = self.coor - self.box_center

    def cal_radiation(self, theta, phi):
        rcospsi = self.coor[:, :, 0] * np.sin(theta) * np.cos(phi) + self.coor[:, :, 1] * np.sin(theta) * np.sin(phi) + self.coor[:, :, 2] * np.cos(theta)
        ejkrcospsi = np.exp(1j * self.k * rcospsi)
        self.Ntheta = self.dx * self.dy * np.sum((self.DFT_Jx * np.cos(theta) * np.cos(phi) + self.DFT_Jy * np.cos(theta) * np.sin(phi)) * ejkrcospsi)
        self.Nphi = self.dx * self.dy * np.sum((- self.DFT_Jx * np.sin(phi) + self.DFT_Jy * np.cos(phi)) * ejkrcospsi)
        self.Ltheta = self.dx * self.dy * np.sum((self.DFT_Mx * np.cos(theta) * np.cos(phi) + self.DFT_My * np.cos(theta) * np.sin(phi)) * ejkrcospsi)
        self.Lphi = self.dx * self.dy * np.sum((- self.DFT_Mx * np.sin(phi) + self.DFT_My * np.cos(phi)) * ejkrcospsi)
        self.Etheta = - self.A * (self.Lphi + self.eta0 * self.Ntheta)
        self.EPhi = self.A * (self.Ltheta - self.eta0 * self.Nphi)

        return self.Etheta, self.EPhi

class Top(Box):
    def __init__(self):
        super().__init__(xmin, ymin, zmin, xmax, ymax, zmax, freq, out_path)
        # 不规范,box为父类的实例化对象
        self.Ex = box.top[:, :, :, :, 0]
        self.Ey = box.top[:, :, :, :, 1]
        self.Hx = box.top[:, :, :, :, 2]
        self.Hy = box.top[:, :, :, :, 3]

    def cal_EandH(self):
        self.Ex = (self.Ex[1:-1, 1:-1, -1] + self.Ex[1:-1, 2:, -1]) / 2
        self.Ey = (self.Ey[1:-1, 1:-1, -1] + self.Ey[2:, 1:-1, -1]) / 2
        self.Hx = (self.Hx[1:-1, 1:-1, -1] + self.Hx[2:, 1:-1, -1] + self.Hx[1:-1, 1:-1, -2] + self.Hx[2:, 1:-1, -2]) / 4
        self.Hy = (self.Hy[1:-1, 1:-1, -1] + self.Hy[1:-1, 2:, -1] + self.Hy[1:-1, 1:-1, -2] + self.Hy[1:-1, 2:, -2]) / 4

    def cal_JandM(self):
        self.Jx = - self.Hy
        self.Jy = self.Hx
        self.Mx = self.Ey
        self.My = - self.Ex
        del self.Ex, self.Ey, self.Hx, self.Hy

    def DFT(self):
        self.DFT_Jx = np.zeros((self.nx - 2, self.ny - 2), dtype=complex)
        self.DFT_Jy = np.zeros((self.nx - 2, self.ny - 2), dtype=complex)
        self.DFT_Mx = np.zeros((self.nx - 2, self.ny - 2), dtype=complex)
        self.DFT_My = np.zeros((self.nx - 2, self.ny - 2), dtype=complex)
        for n in range(self.iterations):
            e = np.exp(-1j * 2 * np.pi * self.freq * (n + 1) * self.dt)
            self.DFT_Jx += self.Jx[:, :, n] * e
            self.DFT_Jy += self.Jy[:, :, n] * e
            self.DFT_Mx += self.Mx[:, :, n] * e
            self.DFT_My += self.My[:, :, n] * e
        self.DFT_Jx = self.DFT_Jx * self.dt
        self.DFT_Jy = self.DFT_Jy * self.dt
        self.DFT_Mx = self.DFT_Mx * self.dt
        self.DFT_My = self.DFT_My * self.dt
        del self.Jx, self.Jy, self.Mx, self.My

    def cal_coordinate(self):
        self.coor = np.zeros((self.nx - 2, self.ny - 2, 3))
        # 计算以矩形盒左下角为坐标原点时，各单元面中心的坐标
        for y in range(self.ny - 2):
            for x in range(self.nx - 2):
                self.coor[x, y, :] = (self.dx * (x + 1.5), self.dy * (y + 1.5), self.dz * (self.nz - 1))
        # 计算以矩形盒中点为坐标原点时，各单元面中心的坐标
        self.coor = self.coor - self.box_center

    def cal_radiation(self, theta, phi):
        rcospsi = self.coor[:, :, 0] * np.sin(theta) * np.cos(phi) + self.coor[:, :, 1] * np.sin(theta) * np.sin(phi) + self.coor[:, :, 2] * np.cos(theta)
        ejkrcospsi = np.exp(1j * self.k * rcospsi)
        self.Ntheta = self.dx * self.dy * np.sum((self.DFT_Jx * np.cos(theta) * np.cos(phi) + self.DFT_Jy * np.cos(theta) * np.sin(phi)) * ejkrcospsi)
        self.Nphi = self.dx * self.dy * np.sum((- self.DFT_Jx * np.sin(phi) + self.DFT_Jy * np.cos(phi)) * ejkrcospsi)
        self.Ltheta = self.dx * self.dy * np.sum((self.DFT_Mx * np.cos(theta) * np.cos(phi) + self.DFT_My * np.cos(theta) * np.sin(phi)) * ejkrcospsi)
        self.Lphi = self.dx * self.dy * np.sum((- self.DFT_Mx * np.sin(phi) + self.DFT_My * np.cos(phi)) * ejkrcospsi)
        self.Etheta = - self.A * (self.Lphi + self.eta0 * self.Ntheta)
        self.EPhi = self.A * (self.Ltheta - self.eta0 * self.Nphi)

        return self.Etheta, self.EPhi

class Front(Box):
    def __init__(self):
        super().__init__(xmin, ymin, zmin, xmax, ymax, zmax, freq, out_path)
        # 不规范,box为父类的实例化对象
        self.Ex = box.front[:, :, :, :, 0]
        self.Ez = box.front[:, :, :, :, 1]
        self.Hx = box.front[:, :, :, :, 2]
        self.Hz = box.front[:, :, :, :, 3]

    def cal_EandH(self):
        self.Ex = (self.Ex[1:-1, 1, 1:-1] + self.Ex[1:-1, 1, 2:]) / 2
        self.Ez = (self.Ez[1:-1, 1, 1:-1] + self.Ez[2:, 1, 1:-1]) / 2
        self.Hx = (self.Hx[1:-1, 0, 1:-1] + self.Hx[2:, 0, 1:-1] + self.Hx[1:-1, 1, 1:-1] + self.Hx[2:, 1, 1:-1]) / 4
        self.Hz = (self.Hz[1:-1, 0, 1:-1] + self.Hz[1:-1, 0, 2:] + self.Hz[1:-1, 1, 1:-1] + self.Hz[1:-1, 1, 2:]) / 4

    def cal_JandM(self):
        self.Jx = - self.Hz
        self.Jz = self.Hx
        self.Mx = self.Ez
        self.Mz = - self.Ex
        del self.Ex, self.Ez, self.Hx, self.Hz

    def DFT(self):
        self.DFT_Jx = np.zeros((self.nx - 2, self.nz - 2), dtype=complex)
        self.DFT_Jz = np.zeros((self.nx - 2, self.nz - 2), dtype=complex)
        self.DFT_Mx = np.zeros((self.nx - 2, self.nz - 2), dtype=complex)
        self.DFT_Mz = np.zeros((self.nx - 2, self.nz - 2), dtype=complex)
        for n in range(self.iterations):
            e = np.exp(-1j * 2 * np.pi * self.freq * (n + 1) * self.dt)
            self.DFT_Jx += self.Jx[:, :, n] * e
            self.DFT_Jz += self.Jz[:, :, n] * e
            self.DFT_Mx += self.Mx[:, :, n] * e
            self.DFT_Mz += self.Mz[:, :, n] * e
        self.DFT_Jx = self.DFT_Jx * self.dt
        self.DFT_Jz = self.DFT_Jz * self.dt
        self.DFT_Mx = self.DFT_Mx * self.dt
        self.DFT_Mz = self.DFT_Mz * self.dt
        del self.Jx, self.Jz, self.Mx, self.Mz

    def cal_coordinate(self):
        self.coor = np.zeros((self.nx - 2, self.nz - 2, 3))
        # 计算以矩形盒左下角为坐标原点时，各单元面中心的坐标
        for z in range(self.nz - 2):
            for x in range(self.nx - 2):
                self.coor[x, z, :] = (self.dx * (x + 1.5), self.dy, self.dz * (z + 1.5))
        # 计算以矩形盒中点为坐标原点时，各单元面中心的坐标
        self.coor = self.coor - self.box_center

    def cal_radiation(self, theta, phi):
        rcospsi = self.coor[:, :, 0] * np.sin(theta) * np.cos(phi) + self.coor[:, :, 1] * np.sin(theta) * np.sin(phi) + self.coor[:, :, 2] * np.cos(theta)
        ejkrcospsi = np.exp(1j * self.k * rcospsi)
        self.Ntheta = self.dx * self.dz * np.sum((self.DFT_Jx * np.cos(theta) * np.cos(phi) - self.DFT_Jz * np.sin(theta)) * ejkrcospsi)
        self.Nphi = self.dx * self.dz * np.sum((- self.DFT_Jx * np.sin(phi) * ejkrcospsi))
        self.Ltheta = self.dx * self.dz * np.sum((self.DFT_Mx * np.cos(theta) * np.cos(phi) - self.DFT_Mz * np.sin(theta) * ejkrcospsi))
        self.Lphi = self.dx * self.dz * np.sum((- self.DFT_Mx * np.sin(phi)) * ejkrcospsi)
        self.Etheta = - self.A * (self.Lphi + self.eta0 * self.Ntheta)
        self.EPhi = self.A * (self.Ltheta - self.eta0 * self.Nphi)

        return self.Etheta, self.EPhi

class Back(Box):
    def __init__(self):
        super().__init__(xmin, ymin, zmin, xmax, ymax, zmax, freq, out_path)
        # 不规范,box为父类的实例化对象
        self.Ex = box.back[:, :, :, :, 0]
        self.Ez = box.back[:, :, :, :, 1]
        self.Hx = box.back[:, :, :, :, 2]
        self.Hz = box.back[:, :, :, :, 3]

    def cal_EandH(self):
        self.Ex = (self.Ex[1:-1, -1, 1:-1] + self.Ex[1:-1, -1, 2:]) / 2
        self.Ez = (self.Ez[1:-1, -1, 1:-1] + self.Ez[2:, -1, 1:-1]) / 2
        self.Hx = (self.Hx[1:-1, -1, 1:-1] + self.Hx[2:, -1, 1:-1] + self.Hx[1:-1, -2, 1:-1] + self.Hx[2:, -2, 1:-1]) / 4
        self.Hz = (self.Hz[1:-1, -1, 1:-1] + self.Hz[1:-1, -1, 2:] + self.Hz[1:-1, -2, 1:-1] + self.Hz[1:-1, -2, 2:]) / 4

    def cal_JandM(self):
        self.Jx = self.Hz
        self.Jz = - self.Hx
        self.Mx = - self.Ez
        self.Mz = self.Ex
        del self.Ex, self.Ez, self.Hx, self.Hz

    def DFT(self):
        self.DFT_Jx = np.zeros((self.nx - 2, self.nz - 2), dtype=complex)
        self.DFT_Jz = np.zeros((self.nx - 2, self.nz - 2), dtype=complex)
        self.DFT_Mx = np.zeros((self.nx - 2, self.nz - 2), dtype=complex)
        self.DFT_Mz = np.zeros((self.nx - 2, self.nz - 2), dtype=complex)
        for n in range(self.iterations):
            e = np.exp(-1j * 2 * np.pi * self.freq * (n + 1) * self.dt)
            self.DFT_Jx += self.Jx[:, :, n] * e
            self.DFT_Jz += self.Jz[:, :, n] * e
            self.DFT_Mx += self.Mx[:, :, n] * e
            self.DFT_Mz += self.Mz[:, :, n] * e
        self.DFT_Jx = self.DFT_Jx * self.dt
        self.DFT_Jz = self.DFT_Jz * self.dt
        self.DFT_Mx = self.DFT_Mx * self.dt
        self.DFT_Mz = self.DFT_Mz * self.dt
        del self.Jx, self.Jz, self.Mx, self.Mz

    def cal_coordinate(self):
        self.coor = np.zeros((self.nx - 2, self.nz - 2, 3))
        # 计算以矩形盒左下角为坐标原点时，各单元面中心的坐标
        for z in range(self.nz - 2):
            for x in range(self.nx - 2):
                self.coor[x, z, :] = (self.dx * (x + 1.5), self.dy * (self.ny - 1), self.dz * (z + 1.5))
        # 计算以矩形盒中点为坐标原点时，各单元面中心的坐标
        self.coor = self.coor - self.box_center

    def cal_radiation(self, theta, phi):
        rcospsi = self.coor[:, :, 0] * np.sin(theta) * np.cos(phi) + self.coor[:, :, 1] * np.sin(theta) * np.sin(phi) + self.coor[:, :, 2] * np.cos(theta)
        ejkrcospsi = np.exp(1j * self.k * rcospsi)
        self.Ntheta = self.dx * self.dz * np.sum((self.DFT_Jx * np.cos(theta) * np.cos(phi) - self.DFT_Jz * np.sin(theta)) * ejkrcospsi)
        self.Nphi = self.dx * self.dz * np.sum((- self.DFT_Jx * np.sin(phi) * ejkrcospsi))
        self.Ltheta = self.dx * self.dz * np.sum((self.DFT_Mx * np.cos(theta) * np.cos(phi) - self.DFT_Mz * np.sin(theta) * ejkrcospsi))
        self.Lphi = self.dx * self.dz * np.sum((- self.DFT_Mx * np.sin(phi)) * ejkrcospsi)
        self.Etheta = - self.A * (self.Lphi + self.eta0 * self.Ntheta)
        self.EPhi = self.A * (self.Ltheta - self.eta0 * self.Nphi)

        return self.Etheta, self.EPhi

class Left(Box):
    def __init__(self):
        super().__init__(xmin, ymin, zmin, xmax, ymax, zmax, freq, out_path)
        # 不规范,box为父类的实例化对象
        self.Ey = box.left[:, :, :, :, 0]
        self.Ez = box.left[:, :, :, :, 1]
        self.Hy = box.left[:, :, :, :, 2]
        self.Hz = box.left[:, :, :, :, 3]

    def cal_EandH(self):
        self.Ey = (self.Ey[1, 1:-1, 1:-1] + self.Ey[1, 1:-1, 2:]) / 2
        self.Ez = (self.Ez[1, 1:-1, 1:-1] + self.Ez[1, 2:, 1:-1]) / 2
        self.Hy = (self.Hy[0, 1:-1, 1:-1] + self.Hy[0, 2:, 1:-1] + self.Hy[1, 1:-1, 1:-1] + self.Hy[1, 2:, 1:-1]) / 4
        self.Hz = (self.Hz[0, 1:-1, 1:-1] + self.Hz[0, 1:-1, 2:] + self.Hz[1, 1:-1, 1:-1] + self.Hz[1, 1:-1, 2:]) / 4

    def cal_JandM(self):
        self.Jy = self.Hz
        self.Jz = - self.Hy
        self.My = - self.Ez
        self.Mz = self.Ey
        del self.Ey, self.Ez, self.Hy, self.Hz

    def DFT(self):
        self.DFT_Jx = np.zeros((self.ny - 2, self.nz - 2), dtype=complex)
        self.DFT_Jz = np.zeros((self.ny - 2, self.nz - 2), dtype=complex)
        self.DFT_Mx = np.zeros((self.ny - 2, self.nz - 2), dtype=complex)
        self.DFT_Mz = np.zeros((self.ny - 2, self.nz - 2), dtype=complex)
        for n in range(self.iterations):
            e = np.exp(-1j * 2 * np.pi * self.freq * (n + 1) * self.dt)
            self.DFT_Jx += self.Jy[:, :, n] * e
            self.DFT_Jz += self.Jz[:, :, n] * e
            self.DFT_Mx += self.My[:, :, n] * e
            self.DFT_Mz += self.Mz[:, :, n] * e
        self.DFT_Jy = self.DFT_Jx * self.dt
        self.DFT_Jz = self.DFT_Jz * self.dt
        self.DFT_My = self.DFT_Mx * self.dt
        self.DFT_Mz = self.DFT_Mz * self.dt
        del self.Jy, self.Jz, self.My, self.Mz

    def cal_coordinate(self):
        self.coor = np.zeros((self.ny - 2, self.nz - 2, 3))
        # 计算以矩形盒左下角为坐标原点时，各单元面中心的坐标
        for z in range(self.nz - 2):
            for y in range(self.ny - 2):
                self.coor[y, z, :] = (self.dx, self.dy * (y + 1.5), self.dz * (z + 1.5))
        # 计算以矩形盒中点为坐标原点时，各单元面中心的坐标
        self.coor = self.coor - self.box_center

    def cal_radiation(self, theta, phi):
        rcospsi = self.coor[:, :, 0] * np.sin(theta) * np.cos(phi) + self.coor[:, :, 1] * np.sin(theta) * np.sin(phi) + self.coor[:, :, 2] * np.cos(theta)
        ejkrcospsi = np.exp(1j * self.k * rcospsi)
        self.Ntheta = self.dy * self.dz * np.sum((self.DFT_Jy * np.cos(theta) * np.sin(phi) - self.DFT_Jz * np.sin(theta)) * ejkrcospsi)
        self.Nphi = self.dy * self.dz * np.sum((self.DFT_Jy * np.cos(phi) * ejkrcospsi))
        self.Ltheta = self.dy * self.dz * np.sum((self.DFT_My * np.cos(theta) * np.sin(phi) - self.DFT_Mz * np.sin(theta) * ejkrcospsi))
        self.Lphi = self.dy * self.dz * np.sum((self.DFT_My * np.cos(phi)) * ejkrcospsi)
        self.Etheta = - self.A * (self.Lphi + self.eta0 * self.Ntheta)
        self.EPhi = self.A * (self.Ltheta - self.eta0 * self.Nphi)

        return self.Etheta, self.EPhi

class Right(Box):
    def __init__(self):
        super().__init__(xmin, ymin, zmin, xmax, ymax, zmax, freq, out_path)
        # 不规范,box为父类的实例化对象
        self.Ey = box.right[:, :, :, :, 0]
        self.Ez = box.right[:, :, :, :, 1]
        self.Hy = box.right[:, :, :, :, 2]
        self.Hz = box.right[:, :, :, :, 3]

    def cal_EandH(self):
        self.Ey = (self.Ey[-1, 1:-1, 1:-1] + self.Ey[-1, 1:-1, 2:]) / 2
        self.Ez = (self.Ez[-1, 1:-1, 1:-1] + self.Ez[-1, 2:, 1:-1]) / 2
        self.Hy = (self.Hy[-1, 1:-1, 1:-1] + self.Hy[-1, 2:, 1:-1] + self.Hy[-2, 1:-1, 1:-1] + self.Hy[-2, 2:, 1:-1]) / 4
        self.Hz = (self.Hz[-1, 1:-1, 1:-1] + self.Hz[-1, 1:-1, 2:] + self.Hz[-2, 1:-1, 1:-1] + self.Hz[-2, 1:-1, 2:]) / 4

    def cal_JandM(self):
        self.Jy = - self.Hz
        self.Jz = self.Hy
        self.My = self.Ez
        self.Mz = - self.Ey
        del self.Ey, self.Ez, self.Hy, self.Hz

    def DFT(self):
        self.DFT_Jx = np.zeros((self.ny - 2, self.nz - 2), dtype=complex)
        self.DFT_Jz = np.zeros((self.ny - 2, self.nz - 2), dtype=complex)
        self.DFT_Mx = np.zeros((self.ny - 2, self.nz - 2), dtype=complex)
        self.DFT_Mz = np.zeros((self.ny - 2, self.nz - 2), dtype=complex)
        for n in range(self.iterations):
            e = np.exp(-1j * 2 * np.pi * self.freq * (n + 1) * self.dt)
            self.DFT_Jx += self.Jy[:, :, n] * e
            self.DFT_Jz += self.Jz[:, :, n] * e
            self.DFT_Mx += self.My[:, :, n] * e
            self.DFT_Mz += self.Mz[:, :, n] * e
        self.DFT_Jy = self.DFT_Jx * self.dt
        self.DFT_Jz = self.DFT_Jz * self.dt
        self.DFT_My = self.DFT_Mx * self.dt
        self.DFT_Mz = self.DFT_Mz * self.dt
        del self.Jy, self.Jz, self.My, self.Mz

    def cal_coordinate(self):
        self.coor = np.zeros((self.ny - 2, self.nz - 2, 3))
        # 计算以矩形盒左下角为坐标原点时，各单元面中心的坐标
        for z in range(self.nz - 2):
            for y in range(self.ny - 2):
                self.coor[y, z, :] = (self.dx * (self.nx - 1), self.dy * (y + 1.5), self.dz * (z + 1.5))
        # 计算以矩形盒中点为坐标原点时，各单元面中心的坐标
        self.coor = self.coor - self.box_center

    def cal_radiation(self, theta, phi):
        rcospsi = self.coor[:, :, 0] * np.sin(theta) * np.cos(phi) + self.coor[:, :, 1] * np.sin(theta) * np.sin(phi) + self.coor[:, :, 2] * np.cos(theta)
        ejkrcospsi = np.exp(1j * self.k * rcospsi)
        self.Ntheta = self.dy * self.dz * np.sum((self.DFT_Jy * np.cos(theta) * np.sin(phi) - self.DFT_Jz * np.sin(theta)) * ejkrcospsi)
        self.Nphi = self.dy * self.dz * np.sum((self.DFT_Jy * np.cos(phi) * ejkrcospsi))
        self.Ltheta = self.dy * self.dz * np.sum((self.DFT_My * np.cos(theta) * np.sin(phi) - self.DFT_Mz * np.sin(theta) * ejkrcospsi))
        self.Lphi = self.dy * self.dz * np.sum((self.DFT_My * np.cos(phi)) * ejkrcospsi)
        self.Etheta = - self.A * (self.Lphi + self.eta0 * self.Ntheta)
        self.EPhi = self.A * (self.Ltheta - self.eta0 * self.Nphi)

        return self.Etheta, self.EPhi




start = time.time()
########################################
out_path = r''
pattern_path = r''
freq = 0.95e9
xmin, ymin, zmin, xmax, ymax, zmax = 23,23,23,26,26,176

box = Box(xmin, ymin, zmin, xmax, ymax, zmax, freq, out_path)
box.generate_index_array()
box.read_from_out()
box.cal_pattern(pattern_path)
##########################################

end = time.time()
print('Running time: %s Seconds'%(end-start))

pass
