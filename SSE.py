import numpy as np
from itertools import product
from numba import njit

qe = 1      # electron charge

class two_term_device:

    def __init__(self, Vth, control):
        self.Vth = Vth
        self.control = control

    def lf(self, V, ctrl):
        """Returns the fixed-voltage controlled forward Poisson rate
        for given V (ctrl is a list of control voltages)"""
        return

    def lr(self, V, ctrl):
        """Returns the fixed-voltage controlled reverse Poisson rate
        for given V (ctrl is a list of control voltages)"""
        return


class shockley_diode(two_term_device):

    def __init__(self, I0, n, Vth):

        super().__init__(Vth,[])
        self.I0 = I0
        self.n = n

    def lf(self, V, ctrl=[]):

        lf = (I_0/qe)*np.exp(V/(self.n*self.Vth))

        return lf

    def lr(self, V, ctrl=[]):

        lr = I_0/qe

        return lr


class tunnel_junction(two_term_device):

    def __init__(self, R, Vth):

        super().__init__(Vth, [])
        self.R = R

    def lf(self, V, ctrl=[]):

        V += 1e-9
        lf = (V/(qe*self.R))/(1-np.exp(-V/self.Vth))

        return lf

    def lr(self, V, ctrl=[]):

        V += 1e-9
        lr = (V/(qe*self.R))/(-1+np.exp(V/self.Vth))

        return lr


class mosfet(two_term_device):

    def __init__(self, I0, VT, n, Vth, control):

        super().__init__(Vth, control)
        self.I0 = I0
        self.VT = VT
        self.n = n

        self.lfactor = (self.I0/qe)*np.exp(-self.VT/(self.n*self.Vth))

    def lf(self, V, ctrl):

        return mosfet.__lf(self.lfactor, ctrl[0], self.n, self.Vth)

    def lr(self, V, ctrl):

        return mosfet.__lr(self.lfactor, ctrl[0], self.n, self.Vth, V)

    @staticmethod
    @njit
    def __lf(lfactor, Vg, n, Vth):
        return lfactor*np.exp(Vg/(n*Vth))

    @staticmethod
    @njit
    def __lr(lfactor, Vg, n, Vth, V):
        return lfactor*np.exp(Vg/(n*Vth))*np.exp(-V/Vth)


class mosfet_free_body(two_term_device):

    def __init__(self, I0, VT, n, Vth, control):

        super().__init__(Vth, control)
        self.I0 = I0
        self.VT = VT
        self.n = n

        self.lfactor = (self.I0/qe)*np.exp(-self.VT/(self.n*self.Vth))

    def lf(self, V, ctrl):

        return mosfet_free_body.__lf(self.lfactor, ctrl[0], ctrl[1], self.n, self.Vth, V)

    def lr(self, V, ctrl):

        return mosfet_free_body.__lr(self.lfactor, ctrl[0], ctrl[2], self.n, self.Vth, V)

    @staticmethod
    @njit
    def __lf(lfactor, Vg, Vd, n, Vth, V):
        return lfactor*np.exp(Vg/(n*Vth))*np.exp(-Vd/Vth)*np.exp(V/Vth)

    @staticmethod
    @njit
    def __lr(lfactor, Vg, Vs, n, Vth, V):
        return lfactor*np.exp(Vg/(n*Vth))*np.exp(-Vs/Vth)*np.exp(-V/Vth)


class circuit:

    def __init__(self, cap, dev, src):

        self.cap = cap
        self.dev = dev
        self.src = src

        self.__set_N()
        self.__build_C_matrix()
        self.__split_C()
        self.__build_Delta()

        return

    def __set_N(self):
        """Obtains the number of free and regulated conductors from the circuit
        description"""

        N = 0
        for c in self.cap:
            m = max(c[0])
            if m > N:
                N = m

        for d in self.dev:
            m = max(d[0])
            if m > N:
                N = m

        for r in self.src:
            if r[0] > N:
                N = r[0]

        self.N = N
        self.Nr = len(self.src)
        self.Nf = self.N - self.Nr

        return

    def __build_C_matrix(self):
        """Contruct the lumped and Maxwell capacitance matrices"""

        self.lumped_C = np.zeros((self.N,self.N))

        for c in self.cap:
            self.lumped_C[c[0][0]-1,c[0][1]-1] += c[1]
            self.lumped_C[c[0][1]-1,c[0][0]-1] += c[1]

        self.maxwell_C = -1*np.copy(self.lumped_C)
        for n in range(self.N):
            self.maxwell_C[n,n] = np.sum(self.lumped_C[:,n])

        self.lumped_C = self.lumped_C
        self.maxwell_C = self.maxwell_C

        return

    def __split_C(self):
        """Split the capacitance matrix into blocks corresponding to free
        and regulated conductors"""

        self.Vr = []
        self.reg_idx = []
        for s in self.src:
            self.reg_idx.append(s[0]-1)
            self.Vr.append(s[1])

        #self.Vr = np.array(self.Vr).reshape(self.Nr,1)

        self.free_idx = [k for k in range(self.N) if k not in self.reg_idx]

        self.Cr = self.maxwell_C[self.reg_idx,:][:,self.reg_idx]
        self.C = self.maxwell_C[self.free_idx,:][:,self.free_idx]
        self.Cx = self.maxwell_C[self.free_idx,:][:,self.reg_idx]

        self.invC = np.linalg.inv(self.C)

        return

    def __build_Delta(self):
        """Builds the incidence matrix of the circuit"""

        ndev = len(self.dev)

        self.Delta = np.zeros((self.N, ndev), dtype=np.int64)

        for k in range(ndev):
            self.Delta[self.dev[k][0][0]-1, k] = -1
            self.Delta[self.dev[k][0][1]-1, k] = 1

        self.free_Delta = self.Delta[self.free_idx,:]
        self.reg_Delta = self.Delta[self.reg_idx,:]

        return

    def update_C(self):
        """Update the capacitance values"""

        self.__build_C_matrix()
        self.__split_C()

        return

    def update_Vr(self):
        """Update the regulated voltages"""

        for k, s in zip(range(self.Nr), self.src):
            self.Vr[k] = s[1]

        return

    def V(self, q, t):
        """Returns the voltage of all conductors for a given state and time"""

        Vr = np.zeros((len(self.Vr),1))
        for k in range(len(self.Vr)):
            if callable(self.Vr[k]):
                Vr[k] = self.Vr[k](t)
            else:
                Vr[k] = self.Vr[k]

        Vf = self.invC.dot(q-self.Cx.dot(Vr))

        V = np.zeros((self.N,1))
        V[self.free_idx] = Vf
        V[self.reg_idx] = Vr

        return V

    def dV(self, dev_idx):
        """Return the change in voltage of all conductors for the forward
        and reverse transitions of a given device (the change in the
        voltage of regulared conductors is just set to zero)"""

        delta = self.free_Delta[:, dev_idx:dev_idx+1]

        dVf = qe*self.invC.dot(delta)
        dV = np.zeros((self.N,1))
        dV[self.free_idx] = dVf

        return dV

@njit
def av_deltaV(V, dV, dev_term, control_a, control_b, t):

    av_V_f = V + dV/2
    av_V_r = V - dV/2

    av_DV_f = av_V_f[dev_term[0]-1] - av_V_f[dev_term[1]-1]
    av_DV_r = av_V_r[dev_term[0]-1] - av_V_r[dev_term[1]-1]

    av_DV_ctrl_f = []
    av_DV_ctrl_r = []

    for ctrl_a, ctrl_b in zip(control_a, control_b):

        av_DV_control_f = av_V_f[ctrl_a-1] - av_V_f[ctrl_b-1]
        av_DV_control_r = av_V_r[ctrl_a-1] - av_V_r[ctrl_b-1]

        av_DV_ctrl_f.append(av_DV_control_f)
        av_DV_ctrl_r.append(av_DV_control_r)

    return av_DV_f, av_DV_r, av_DV_ctrl_f, av_DV_ctrl_r


class simulation:

    def __init__(self, circuit):

        self.circuit = circuit

        self.rates_f = np.zeros((len(self.circuit.dev),1))
        self.rates_r = np.zeros_like(self.rates_f)

        self.av_dv_f = np.zeros_like(self.rates_f)
        self.av_dv_r = np.zeros_like(self.rates_f)

        self.controls_a = []
        self.controls_b = []
        for dev in self.circuit.dev:
            self.controls_a.append(np.array([c[0] for c in dev[1].control], dtype=np.int64))
            self.controls_b.append(np.array([c[1] for c in dev[1].control], dtype=np.int64))

        self.Ndev = len(self.circuit.dev)
        self.dV = []
        for d in range(self.Ndev):
            self.dV.append(self.circuit.dV(d))

        return

    def run_tl(self, dt, tf, init_st = None, max_iter = None):

        np.random.seed()

        self.dt = dt
        self.tf = tf
        self.times = np.arange(0,tf,dt)
        self.traj = np.zeros((self.circuit.Nf, len(self.times)), dtype=np.int64)
        self.jumps = np.zeros((len(self.circuit.dev), len(self.times)-1), dtype=np.int64)
        self.Q = np.zeros((len(self.circuit.dev), len(self.times)-1))

        if init_st != None:
            self.traj[:,0] = init_st

        for s in range(len(self.times)-1):
            dq, Nj, Q = self.__update_tau_leaping(self.traj[:,s:s+1], self.times[s])
            self.traj[:,s+1:s+2] = self.traj[:,s:s+1] + dq
            self.jumps[:,s:s+1] = Nj
            self.Q[:,s:s+1] = Q

        return

    def __update_tau_leaping(self, q, t):

        V = self.circuit.V(q,t)

        for d in range(self.Ndev):

            dV = self.dV[d]
            dev_term = self.circuit.dev[d][0]
            dev = self.circuit.dev[d][1]

            av_DV_f, av_DV_r, ctrl_f, ctrl_r = \
            av_deltaV(V, dV, dev_term, self.controls_a[d], self.controls_b[d], t)

            self.rates_f[d] = dev.lf(av_DV_f, ctrl_f)
            self.rates_r[d] = dev.lr(av_DV_r, ctrl_r)

            self.av_dv_f[d] = av_DV_f
            self.av_dv_r[d] = av_DV_r

        Nf = np.random.poisson(self.rates_f*self.dt)
        Nr = np.random.poisson(self.rates_r*self.dt)
        Nj = Nf - Nr

        dq = self.circuit.free_Delta.dot(Nj) # charge change
        Q = Nf*self.av_dv_f - Nr*self.av_dv_r # heat

        return dq, Nj, Q


    def run_gill(self, tf, init_st = None, max_iter = None):

        np.random.seed()

        self.tf = tf
        self.times = [0]
        self.traj = []
        self.jumps = []

        if init_st != None:
            self.traj.append(init_st)
        else:
            self.traj.append(np.zeros((self.circuit.Nf, 1)))

        while self.times[-1] < self.tf:
            dt, dq, jump = self.__update_gillespie(self.traj[-1], self.times[-1])
            self.times.append(self.times[-1] + dt)
            self.traj.append(self.traj[-1] + dq)
            self.jumps.append(jump)

        return

    def __update_gillespie(self, q, t):

        V = self.circuit.V(q,t)

        for d in range(self.Ndev):

            dV = self.dV[d]
            dev_term = self.circuit.dev[d][0]
            dev = self.circuit.dev[d][1]

            av_DV_f, av_DV_r, ctrl_f, ctrl_r = \
            av_deltaV(V, dV, dev_term, self.controls_a[d], self.controls_b[d], t)

            self.rates_f[d] = dev.lf(av_DV_f, ctrl_f)
            self.rates_r[d] = dev.lr(av_DV_r, ctrl_r)

        dt_f = np.random.exponential(1/self.rates_f)
        dt_r = np.random.exponential(1/self.rates_r)

        amin_dt_f = np.argmin(dt_f)
        amin_dt_r = np.argmin(dt_r)

        if dt_f[amin_dt_f] <= dt_r[amin_dt_r]:
            jump = amin_dt_f
            dq = self.circuit.free_Delta[:, amin_dt_f:amin_dt_f+1] # charge change
            dt = dt_f[amin_dt_f]
        else:
            jump = -amin_dt_r
            dq = -self.circuit.free_Delta[:, amin_dt_r:amin_dt_r+1] # charge change
            dt = dt_r[amin_dt_r]

        return dt, dq, jump#, Q
