"""
standalone implementation adaptation of models from TVB 
so license is GPL w/ the notice "(c) 2012-2025, Baycrest
Centre for Geriatric Care ("Baycrest") and others".

adaptations include:

- removal of numba & numexpr (not available in browser)
- mocks/shims for traits
- batching in reduced set models

"""

import numpy
np = numpy

NArray = Range = Final = List = dict


class Model:
    state_variables = ()  # type: typing.Tuple[str]
    non_integrated_variables = None  # type: typing.Tuple[str]
    variables_of_interest = ()
    _nvar = None
    _nintvar = _nvar
    nmode = number_of_modes = 1
    cvar = None
    stvar = None
    state_variable_boundaries = None
    state_variable_mask = None

    def __init__(self, **kwargs):
        for key in dir(self):
            value = getattr(self, key)
            if isinstance(value, dict) and 'default' in value:
                setattr(self, key, kwargs.get(key, value['default']))
        if hasattr(self, 'update_derived_parameters'):
            self.update_derived_parameters()
        self.nvar = self._nvar
        self.nmode = self.number_of_modes

ModelNumbaDfun = Model  # placeholder for numba version

def test_models():
    n_node = 32
    batch_size = 8
    model_count = 0
    for name, obj in globals().items():
        if isinstance(obj, type) and issubclass(obj, Model):
            if obj in (Model, ReducedSetBase, ModelNumbaDfun):
                continue
            m = obj()
            print(f'{model_count} {obj.__name__} ', end='')
            print(f' {m._nvar} {m.cvar.size} {m.nmode}')
            x = np.zeros((m._nvar, n_node, m.nmode, batch_size))
            cx = np.zeros((m.cvar.size, n_node, m.nmode, batch_size))
            m.dfun(x, cx)
            model_count += 1


class KIonEx(Model):
    r"""
    KIonEx (Potassium K+ Ion exchange) mean-field model was developed in (Bandyopadhyay & Rabuffo et al. 2023). 
    It describes the mean-field activity of a population of Hodgkin-Huxley-type neurons (Depannemaker et al 2022) 
    linking the slow fluctuations of intra- and extra-cellular potassium ion concentrations to the mean membrane potential, 
    and the synaptic input to the population firing rate. 
    The model is derived as the mathematical limit of an infinite number of all-to-all coupled neurons, resulting in 5 state variables:
    :math:`x` represents a phenomenological variable connected to the firing rate, 
    :math:`V` represent the average membrane potential,
    :math:`n` represents the gating variable for potassium K, 
    :math:`\Delta K_{int}` represent the intracellular potassium concentration,
    :math:`K_g` represents the extracellular potassium buffering by the external bath
    """
    #_ui_name = "KIonEx"
    #ui_configurable_parameters = ['E', 'K_bath', 'J', 'eta', 'Delta','c_minus','R_minus','c_plus','R_plus','Vstar']

    E = NArray(
        label=r":math:`E`",
        default=np.array([0.]),
        domain=Range(lo=-80, hi=0, step=0.5),
        doc="""Reversal Potential""",
    )

    K_bath = NArray(
        label=r":math:`K_bath`",
        default=np.array([5.5]),
        domain=Range(lo=3, hi=40.0, step=0.25),
        doc="""Potassium concentration in bath""",
    )

    J = NArray(
        label=r":math:`J`",
        default=np.array([0.1]),
        domain=Range(lo=0.001, hi=40.0, step=0.01),
        doc="""Mean Synaptic weight""",
    )

    eta = NArray(
        label=":math:`eta`",
        default=np.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.1),
        doc="""Mean heterogeneous noise""",
    )

    Delta = NArray(
        label=r":math:`\Delta`",
        default=np.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""HWHM heterogeneous noise""",
    )

    c_minus = NArray(
        label=":math:`c_minus`",
        default=np.array([-40.0]),
        domain=Range(lo=-100.0, hi=-10.0, step=0.5),
        doc="""x-coordinate left parabola""",
    )

    R_minus = NArray(
        label=r":math:`R_minus`",
        default=np.array([0.5]),
        domain=Range(lo=0.0001, hi=5.0, step=0.01),
        doc="""curvature left parabola""",
    )

    c_plus = NArray(
        label=":math:`c_plus`",
        default=np.array([-20.0]),
        domain=Range(lo=-80.0, hi=0.0, step=0.5),
        doc="""x-coordinate right parabola""",
    )

    R_plus = NArray(
        label=r":math:`R_plus`",
        default=np.array([-0.5]),
        domain=Range(lo=-5.0, hi=-0.0001, step=0.01),
        doc="""curvature right parabola""",
    )


    Vstar = NArray(
        label=r":math:`Vstar`",
        default=np.array([-31]),
        domain=Range(lo=-55.0, hi=-15, step=0.5),
        doc="""x-coordinate meeting point of parabolas""",
    )

    #'Cm': 1, #nF, # membrane capacitance
    Cm = NArray(
        label=r":math:`Cm`",
        default=np.array([1]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""membrane capacitance""",
    )

    # 'tau_n': 4, # ms # time constant of gating variable
    tau_n = NArray(
        label=r":math:`tau_n`",
        default=np.array([4]),
        domain=Range(lo=2, hi=6, step=0.5),
        doc="""time constant of gating variable""",
    )

    # 'gamma': 0.04,  # mol / C  # conversion factor
    gamma = NArray(
        label=r":math:`gamma`",
        default=np.array([0.04]),
        domain=Range(lo=0.02, hi=0.06, step=0.005),
        doc="""conversion factor""",
    )

    #'epsilon': 0.001, # mHz  # diffusion rate
    epsilon = NArray(
        label=r":math:`epsilon`",
        default=np.array([0.001]),
        domain=Range(lo=0.0005, hi=0.0015, step=0.0001),
        doc="""diffusion rate""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "x": np.array([0.0, np.inf]),
            "V": np.array([-500.0, np.inf]),
            "n": np.array([0.0, np.inf]),
            "DKi": np.array([-100.0, np.inf]),
            "Kg": np.array([-100.0, np.inf])
        },
    )

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "x": np.array([0., 1]),
            "V": np.array([-90., 10.]),
            "n": np.array([0., 1]),
            "DKi": np.array([-10, 0]),
            "Kg": np.array([-20, -5])
        },
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )



    # TODO should match cvars below..
    coupling_terms = Final(
        label="Coupling terms",
        # how to unpack coupling array
        default=["Coupling_Term"]
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("x", "V","n","DKi", "Kg"),
        default=("x", "V","n","DKi", "Kg"),
        doc="The quantities of interest for monitoring for the Infinite HH 5D.",
    )

    state_variables = ['x', 'V','n','DKi','Kg']
    _nvar = 5
    # Cvar is the coupling variable. 
    cvar = np.array([0], dtype=np.int32)
    # Stvar is the variable where stimulus is applied.
    stvar = np.array([1], dtype=np.int32)
    
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The mean-field approximation for a population of Hodgkin-Huxley-type neurons driven by slow potassium dynamics consists of a 5D system:

        .. math::
            \frac{dx}{dt}&=
            \begin{cases} 
            \Delta+2R_{-}(V-c_{-})x - J r x; \  V\leq V^{\star}\\
            \Delta+2R_{+}(V-c_{+})x - J r x; \  V> V^{\star},
            \end{cases}\\
            \frac{dV}{dt}&=
            \begin{cases} 
            -\frac{1}{C_m}(I_{Cl}+I_{Na}+I_{K}+I_{pump})-R_{-}x^2+J r(E_{syn}-V)+\overline{\eta}; \  V\leq V^{\star}\\
            -\frac{1}{C_m}(I_{Cl}+I_{Na}+I_{K}+I_{pump})-R_{+}x^2+J r(E_{syn}-V)+\overline{\eta}; \  V>V^{\star}, 
            \end{cases}\\
            \frac{dn}{dt} &= \frac{n_{\infty}(V)-n}{\tau_n}, \\
            \frac{d \Delta [K^{+}]_{int}}{dt} &= - \frac{\gamma}{\omega_i}(I_K - 2 I_{pump}),\\
            \frac{d[K^+]_g}{dt} &= \epsilon ([K^+]_{bath} - [K^+]_{ext}\}).\\

        For details refer to (Bandyopadhyay & Rabuffo et al. 2023)
        """
        
        x = state_variables[0, :]
        V = state_variables[1, :]
        n = state_variables[2, :]
        DKi = state_variables[3, :]
        Kg = state_variables[4, :]
        
        #[State_variables, nodes]
        E = self.E

        K_bath = self.K_bath
        J = self.J
        eta = self.eta
        Delta = self.Delta

        c_minus = self.c_minus
        R_minus = self.R_minus
        c_plus = self.c_plus
        R_plus = self.R_plus
        Vstar = self.Vstar

        Cm      = self.Cm
        tau_n   = self.tau_n
        gamma   = self.gamma
        epsilon = self.epsilon
     
        Coupling_Term = coupling[0, :] #This zero refers to the first element of cvar (trivial in this case)

        # Constants
        Cnap = 21.0  # mol.m**-3 
        DCnap = 2.0  # mol.m**-3 
        Ckp = 5.5  # mol.m**-3 
        DCkp = 1.0  # mol.m**-3 
        Cmna = -24.0  # mV 
        DCmna = 12.0  # mV 
        Chn = 0.4  # dimensionless 
        DChn = -8.0  # dimensionless 
        Cnk = -19.0  # mV 
        DCnk = 18.0  # mV #Ok in the paper
        g_Cl = 7.5  # nS #Ok in the paper   # chloride conductance
        g_Na = 40.0  # nS   # maximal sodiumconductance
        g_K = 22.0  # nS  # maximal potassium conductance
        g_Nal = 0.02  # nS  # sodium leak conductance
        g_Kl = 0.12  # nS  # potassium leak conductance
        rho = 250.  # 250.,#pA # maximal Na/K pump current
        w_i = 2160.0  # umeter**3  # intracellular volume 
        w_o = 720.0  # umeter**3 # extracellular volume 
        Na_i0 = 16.0  # mMol/m**3 # initial concentration of intracellular Na
        Na_o0 = 138.0 # mMol/m**3 # initial concentration of extracellular Na
        K_i0 = 130.0  # mMol/m**3 # initial concentration of intracellular K
        K_o0 = 4.80   # mMol/m**3 # initial concentration of extracellular K
        Cl_i0 = 5.0   # mMol/m**3 # initial concentration of intracellular Cl
        Cl_o0 = 112.0 # mMol/m**3 # initial concentration of extracellular Cl
        

        # helper functions

        def m_inf(V):
            return 1.0/(1.0+np.exp((Cmna-V)/DCmna))

        def n_inf(V):
            return 1.0/(1.0+np.exp((Cnk-V)/DCnk))

        def h(n):
            return 1.1 - 1.0 / (1.0 + np.exp(-8.0 * (n - 0.4)))

        def I_K_form(V,n,K_o,K_i):
            return (g_Kl+g_K*n)*(V- 26.64*np.log(K_o/K_i)) 

        def I_Na_form(V,Na_o,Na_i,n):
            return (g_Nal+g_Na*m_inf(V)*h(n))*(V- 26.64*np.log(Na_o/Na_i))

        def I_Cl_form(V):
            return g_Cl*(V+ 26.64*np.log(Cl_o0/Cl_i0)) 

        def I_pump_form(Na_i,K_o):
            return rho*(1.0/(1.0+np.exp((Cnap - Na_i) / DCnap))*(1.0/(1.0+np.exp((Ckp - K_o)/DCkp)))) 

        def V_dot_form(I_Na,I_K,I_Cl,I_pump):
            return (-1.0/Cm)*(I_Na+I_K+I_Cl+I_pump) 

        beta= w_i / w_o 
        DNa_i = -DKi 
        DNa_o = -beta * DNa_i
        DK_o = -beta * DKi
        K_i = K_i0 + DKi 
        Na_i = Na_i0 + DNa_i 
        Na_o = Na_o0 + DNa_o 
        K_o = K_o0 + DK_o + Kg 

        ninf=n_inf(V)
        I_K = I_K_form(V,n,K_o,K_i)
        I_Na = I_Na_form(V,Na_o,Na_i,n)
        I_Cl = I_Cl_form(V)
        I_pump = I_pump_form(Na_i,K_o)
    
        r = R_minus*x/np.pi
        Vdot = (-1.0/Cm)*(I_Na+I_K+I_Cl+I_pump) 

        derivative = np.empty_like(state_variables)

        if_xdot = Delta+2*R_minus*(V-c_minus)*x-J*r*x 
        else_xdot = Delta+2*R_plus*(V-c_plus)*x-J*r*x
        derivative[0] = np.where(V <= Vstar, if_xdot, else_xdot)

        if_Vdot = Vdot - R_minus*x**2 + eta + (R_minus/np.pi)*Coupling_Term*(E-V)
        else_Vdot = Vdot - R_plus*x**2 + eta + (R_minus/np.pi)*Coupling_Term*(E-V)
        derivative[1] = np.where(V <= Vstar, if_Vdot, else_Vdot)

        derivative[2] = (ninf - n) / tau_n
        derivative[3] = -(gamma / w_i) * (I_K - 2.0 * I_pump)
        derivative[4] = epsilon * (K_bath - K_o)
        
        return derivative


class Epileptor(Model):
    r"""
    The Epileptor is a composite neural mass model of six dimensions which
    has been crafted to model the phenomenology of epileptic seizures.
    (see [Jirsaetal_2014]_)
    Equations and default parameters are taken from [Jirsaetal_2014]_.
          +------------------------------------------------------+
          |                         Table 1                      |
          +----------------------+-------------------------------+
          |        Parameter     |           Value               |
          +======================+===============================+
          |         I_rest1      |              3.1              |
          +----------------------+-------------------------------+
          |         I_rest2      |              0.45             |
          +----------------------+-------------------------------+
          |         r            |            0.00035            |
          +----------------------+-------------------------------+
          |         x_0          |             -1.6              |
          +----------------------+-------------------------------+
          |         slope        |              0.0              |
          +----------------------+-------------------------------+
          |             Integration parameter                    |
          +----------------------+-------------------------------+
          |           dt         |              0.1              |
          +----------------------+-------------------------------+
          |  simulation_length   |              4000             |
          +----------------------+-------------------------------+
          |                    Noise                             |
          +----------------------+-------------------------------+
          |         nsig         | [0., 0., 0., 1e-3, 1e-3, 0.]  |
          +----------------------+-------------------------------+
          |              Jirsa et al. 2014                       |
          +------------------------------------------------------+
    .. figure :: img/Epileptor_01_mode_0_pplane.svg
        :alt: Epileptor phase plane
    .. [Jirsaetal_2014] Jirsa, V. K.; Stacey, W. C.; Quilichini, P. P.;
        Ivanov, A. I.; Bernard, C. *On the nature of seizure dynamics.* Brain,
        2014.

    Variables of interest to be used by monitors: -y[0] + y[3]

        .. math::
            \dot{x_{1}} &=& y_{1} - f_{1}(x_{1}, x_{2}) - z + I_{ext1} \\
            \dot{y_{1}} &=& c - d x_{1}^{2} - y{1} \\
            \dot{z} &=&
            \begin{cases}
            r(4 (x_{1} - x_{0}) - z-0.1 z^{7}) & \text{if } x<0 \\
            r(4 (x_{1} - x_{0}) - z) & \text{if } x \geq 0
            \end{cases} \\
            \dot{x_{2}} &=& -y_{2} + x_{2} - x_{2}^{3} + I_{ext2} + 0.002 g - 0.3 (z-3.5) \\
            \dot{y_{2}} &=& 1 / \tau (-y_{2} + f_{2}(x_{2}))\\
            \dot{g} &=& -0.01 (g - 0.1 x_{1})
    where:
        .. math::
            f_{1}(x_{1}, x_{2}) =
            \begin{cases}
            a x_{1}^{3} - b x_{1}^2 & \text{if } x_{1} <0\\
            -(slope - x_{2} + 0.6(z-4)^2) x_{1} &\text{if }x_{1} \geq 0
            \end{cases}
    and:
        .. math::
            f_{2}(x_{2}) =
            \begin{cases}
            0 & \text{if } x_{2} <-0.25\\
            a_{2}(x_{2} + 0.25) & \text{if } x_{2} \geq -0.25
            \end{cases}
    Note Feb. 2017: the slow permittivity variable can be modify to account for the time
    difference between interictal and ictal states (see [Proixetal_2014]).
    .. [Proixetal_2014] Proix, T.; Bartolomei, F; Chauvel, P; Bernard, C; Jirsa, V.K. *
        Permittivity coupling across brain regions determines seizure recruitment in
        partial epilepsy.* J Neurosci 2014, 34:15009-21.
    """

    a = NArray(
        label=":math:`a`",
        default=np.array([1.0]),
        doc="Coefficient of the cubic term in the first state variable")

    b = NArray(
        label=":math:`b`",
        default=np.array([3.0]),
        doc="Coefficient of the squared term in the first state variabel")

    c = NArray(
        label=":math:`c`",
        default=np.array([1.0]),
        doc="Additive coefficient for the second state variable, \
        called :math:`y_{0}` in Jirsa paper")

    d = NArray(
        label=":math:`d`",
        default=np.array([5.0]),
        doc="Coefficient of the squared term in the second state variable")

    r = NArray(
        label=":math:`r`",
        domain=Range(lo=0.0, hi=0.001, step=0.00005),
        default=np.array([0.00035]),
        doc="Temporal scaling in the third state variable, \
        called :math:`1/\\tau_{0}` in Jirsa paper")

    s = NArray(
        label=":math:`s`",
        default=np.array([4.0]),
        doc="Linear coefficient in the third state variable")

    x0 = NArray(
        label=":math:`x_0`",
        domain=Range(lo=-3.0, hi=-1.0, step=0.1),
        default=np.array([-1.6]),
        doc="Epileptogenicity parameter")

    Iext = NArray(
        label=":math:`I_{ext}`",
        domain=Range(lo=1.5, hi=5.0, step=0.1),
        default=np.array([3.1]),
        doc="External input current to the first population")

    slope = NArray(
        label=":math:`slope`",
        domain=Range(lo=-16.0, hi=6.0, step=0.1),
        default=np.array([0.]),
        doc="Linear coefficient in the first state variable")

    Iext2 = NArray(
        label=":math:`I_{ext2}`",
        domain=Range(lo=0.0, hi=1.0, step=0.05),
        default=np.array([0.45]),
        doc="External input current to the second population")

    tau = NArray(
        label=r":math:`\tau`",
        default=np.array([10.0]),
        doc="Temporal scaling coefficient in fifth state variable")

    aa = NArray(
        label=":math:`aa`",
        default=np.array([6.0]),
        doc="Linear coefficient in fifth state variable")

    bb = NArray(
        label=":math:`bb`",
        default=np.array([2.0]),
        doc="Linear coefficient of lowpass excitatory coupling in fourth state variable")

    Kvf = NArray(
        label=":math:`K_{vf}`",
        default=np.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc="Coupling scaling on a very fast time scale.")

    Kf = NArray(
        label=":math:`K_{f}`",
        default=np.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc="Correspond to the coupling scaling on a fast time scale.")

    Ks = NArray(
        label=":math:`K_{s}`",
        default=np.array([0.0]),
        domain=Range(lo=-4.0, hi=4.0, step=0.1),
        doc="Permittivity coupling, that is from the fast time scale toward the slow time scale")

    tt = NArray(
        label=":math:`K_{tt}`",
        default=np.array([1.0]),
        domain=Range(lo=0.001, hi=10.0, step=0.001),
        doc="Time scaling of the whole system")

    modification = NArray(
        dtype=bool,
        label=":math:`modification`",
        default=np.array([False]),
        doc="When modification is True, then use nonlinear influence on z. \
        The default value is False, i.e., linear influence.")

    state_variable_range = Final(
        default={
            "x1": np.array([-2., 1.]),
            "y1": np.array([-20., 2.]),
            "z": np.array([2.0, 5.0]),
            "x2": np.array([-2., 0.]),
            "y2": np.array([0., 2.]),
            "g": np.array([-1., 1.])
        },
        label="State variable ranges [lo, hi]",
        doc="Typical bounds on state variables in the Epileptor model.")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('x1', 'y1', 'z', 'x2', 'y2', 'g', 'x2 - x1'),
        default=("x2 - x1", 'z'),
        doc="Quantities of the Epileptor available to monitor.",
    )

    state_variables = ('x1', 'y1', 'z', 'x2', 'y2', 'g')

    _nvar = 6
    cvar = np.array([0, 3], dtype=np.int32)  # should these not be constant Attr's?
    cvar.setflags(write=False)  # todo review this

    def dfun(self, state_variables, coupling, local_coupling=0.0,
                    array=np.array, where=np.where, concat=np.concatenate):

        y = state_variables
        ydot = np.empty_like(state_variables)

        Iext = self.Iext + local_coupling * y[0]
        c_pop1 = coupling[0, :]
        c_pop2 = coupling[1, :]

        # population 1
        if_ydot0 = - self.a * y[0] ** 2 + self.b * y[0]
        else_ydot0 = self.slope - y[3] + 0.6 * (y[2] - 4.0) ** 2
        ydot[0] = self.tt * (y[1] - y[2] + Iext + self.Kvf * c_pop1 + where(y[0] < 0., if_ydot0, else_ydot0) * y[0])
        ydot[1] = self.tt * (self.c - self.d * y[0] ** 2 - y[1])

        # energy
        if_ydot2 = - 0.1 * y[2] ** 7
        else_ydot2 = 0
        if self.modification:
            h = self.x0 + 3. / (1. + np.exp(- (y[0] + 0.5) / 0.1))
        else:
            h = 4 * (y[0] - self.x0) + where(y[2] < 0., if_ydot2, else_ydot2)
        ydot[2] = self.tt * (self.r * (h - y[2] + self.Ks * c_pop1))

        # population 2
        ydot[3] = self.tt * (
                -y[4] + y[3] - y[3] ** 3 + self.Iext2 + self.bb * y[5] - 0.3 * (y[2] - 3.5) + self.Kf * c_pop2)
        if_ydot4 = 0
        else_ydot4 = self.aa * (y[3] + 0.25)
        ydot[4] = self.tt * ((-y[4] + where(y[3] < -0.25, if_ydot4, else_ydot4)) / self.tau)

        # filter
        ydot[5] = self.tt * (-0.01 * (y[5] - 0.1 * y[0]))

        return ydot


class Epileptor2D(Model):
    r"""
        Two-dimensional reduction of the Epileptor.

        .. moduleauthor:: courtiol.julie@gmail.com

        Taking advantage of time scale separation and focusing on the slower time scale,
        the five-dimensional Epileptor reduces to a two-dimensional system (see [Proixetal_2014,
        Proixetal_2017]).

        Note: the slow permittivity variable can be modify to account for the time
        difference between interictal and ictal states (see [Proixetal_2014]).

        Equations and default parameters are taken from [Proixetal_2014]:

        .. math::
            \dot{x_{1,i}} &=& - x_{1,i}^{3} - 2x_{1,i}^{2}  + 1 - z_{i} + I_{ext1,i} \\
            \dot{z_{i}} &=& r(h - z_{i})

        with
        .. math::
            h =
            \begin{cases}
            x_{0} + 3 / (exp((x_{1} + 0.5)/0.1)) & \text{if } modification\\
            4 (x_{1,i} - x_{0}) & \text{else }
            \end{cases}
        References:
            [Proixetal_2014] Proix, T.; Bartolomei, F; Chauvel, P; Bernard, C; Jirsa, V.K. *
            Permittivity coupling across brain regions determines seizure recruitment in
            partial epilepsy.* J Neurosci 2014, 34:15009-21.

            [Proixetal_2017] Proix, T.; Bartolomei, F; Guye, M.; Jirsa, V.K. *Individual brain
            structure and modelling predict seizure propagation.* Brain 2017, 140; 641–654.
    """

    a = NArray(
        label=":math:`a`",
        default=np.array([1.0]),
        doc="Coefficient of the cubic term in the first state-variable.")

    b = NArray(
        label=":math:`b`",
        default=np.array([3.0]),
        doc="Coefficient of the squared term in the first state-variable.")

    c = NArray(
        label=":math:`c`",
        default=np.array([1.0]),
        doc="Additive coefficient for the second state-variable x_{2}, \
        called :math:`y_{0}` in Jirsa paper.")

    d = NArray(
        label=":math:`d`",
        default=np.array([5.0]),
        doc="Coefficient of the squared term in the second state-variable x_{2}.")

    r = NArray(
        label=":math:`r`",
        domain=Range(lo=0.0, hi=0.001, step=0.00005),
        default=np.array([0.00035]),
        doc=r"Temporal scaling in the slow state-variable, \
        called :math:`1\tau_{0}` in Jirsa paper (see class Epileptor).")

    x0 = NArray(
        label=":math:`x_0`",
        domain=Range(lo=-3.0, hi=-1.0, step=0.1),
        default=np.array([-1.6]),
        doc="Epileptogenicity parameter.")

    Iext = NArray(
        label=":math:`I_{ext}`",
        domain=Range(lo=1.5, hi=5.0, step=0.1),
        default=np.array([3.1]),
        doc="External input current to the first state-variable.")

    slope = NArray(
        label=":math:`slope`",
        domain=Range(lo=-16.0, hi=6.0, step=0.1),
        default=np.array([0.]),
        doc="Linear coefficient in the first state-variable.")

    Kvf = NArray(
        label=":math:`K_{vf}`",
        default=np.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc="Coupling scaling on a very fast time scale.")

    Ks = NArray(
        label=":math:`K_{s}`",
        default=np.array([0.0]),
        domain=Range(lo=-4.0, hi=4.0, step=0.1),
        doc="Permittivity coupling, that is from the fast time scale toward the slow time scale.")

    tt = NArray(
        label=":math:`tt`",
        default=np.array([1.0]),
        domain=Range(lo=0.001, hi=1.0, step=0.001),
        doc="Time scaling of the whole system to the system in real time.")

    modification = NArray(
        dtype=bool,
        label=":math:`modification`",
        default=np.array([False]),
        doc="When modification is True, then use nonlinear influence on z. \
        The default value is False, i.e., linear influence.")

    state_variable_range = Final(
        default={"x1": np.array([-2., 1.]), "z": np.array([2.0, 5.0])},
        label="State variable ranges [lo, hi]",
        doc="Typical bounds on state-variables in the Epileptor 2D model.")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('x1', 'z'),
        default=('x1',),
        doc="Quantities of the Epileptor 2D available to monitor.")

    state_variables = ('x1', 'z')

    _nvar = 2
    cvar = np.array([0], dtype=np.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0,
                    array=np.array, where=np.where, concat=np.concatenate):

        y = state_variables
        ydot = np.empty_like(state_variables)

        Iext = self.Iext + local_coupling * y[0]
        c_pop = coupling[0, :]

        # population 1
        if_ydot0 = self.a * y[0] ** 2 + (self.d - self.b) * y[0]
        else_ydot0 = - self.slope - 0.6 * (y[1] - 4.0) ** 2 + self.d * y[0]

        ydot[0] = self.tt * (self.c - y[1] + Iext + self.Kvf * c_pop - (where(y[0] < 0., if_ydot0, else_ydot0)) * y[0])

        # energy
        if_ydot1 = - 0.1 * y[1] ** 7
        else_ydot1 = 0

        if self.modification:
            h = self.x0 + 3. / (1. + np.exp(- (y[0] + 0.5) / 0.1))
        else:
            h = 4 * (y[0] - self.x0) + where(y[1] < 0., if_ydot1, else_ydot1)

        ydot[1] = self.tt * (self.r * (h - y[1] + self.Ks * c_pop))

        return ydot



class MontbrioPazoRoxin(Model):
    r"""
    2D model describing the Ott-Antonsen reduction of infinite all-to-all
    coupled QIF neurons (Theta-neurons) as in [Montbrio_Pazo_Roxin_2015]_.

    The two state variables :math:`r` and :math:`V` represent the average
    firing rate and the average membrane potential of our QIF neurons.

    The equations of the infinite QIF 2D population model read

    .. math::
            \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
            \dot{V} &= 1/\tau (V^2 - \tau^2 \pi^2 r^2 + \eta + J \tau r + I)
    
    Input from the network enters in the :math:`V` variable as 
    :math:`1/\tau(c_r C_r + c_v C_V)` where C is the incomming coupling. In 
    other words, depending on the parameters :math:`c_r`, :math:`c_v` we couple
    the neural masses via the firing rate and/or the membrane potential.
    
    .. [Montbrio_Pazo_Roxin_2015] Montbrió, E., Pazó, D., & Roxin, A. (2015). Macroscopic description for networks of spiking neurons. *Physical Review X*, 5(2), 021028.
    """

    # Define traited attributes for this model, these represent possible kwargs.

    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.001, hi=15.0, step=0.01),
        doc="""Characteristic time""",
    )

    I = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""External Current""",
    )

    Delta = NArray(
        label=r":math:`\Delta`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Mean heterogeneous noise""",
    )

    J = NArray(
        label=":math:`J`",
        default=numpy.array([15.0]),
        domain=Range(lo=-25.0, hi=25.0, step=0.0001),
        doc="""Mean Synaptic weight.""",
    )

    eta = NArray(
        label=r":math:`\eta`",
        default=numpy.array([-5.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the
            firing rate variable to itself""",
    )

    Gamma = NArray(
        label=r":math:`\Gamma`",
        default=numpy.array([0.]),
        domain=Range(lo=0., hi=10.0, step=0.01),
        doc="""Half-width of synaptic weight distribution""",
    )

    cr = NArray(
        label=":math:`cr`",
        default=numpy.array([1.]),
        domain=Range(lo=0., hi=1, step=0.1),
        doc="""It is the weight on Coupling through variable r.""",
    )

    cv = NArray(
        label=":math:`cv`",
        default=numpy.array([0.]),
        domain=Range(lo=0., hi=1, step=0.1),
        doc="""It is the weight on Coupling through variable V.""",
    )

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"r": numpy.array([0., 2.0]),
                 "V": numpy.array([-2.0, 1.5])},
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "r": numpy.array([0.0, numpy.inf])
        },
    )

    # TODO should match cvars below..
    coupling_terms = Final(
        label="Coupling terms",
        # how to unpack coupling array
        default=["Coupling_Term_r", "Coupling_Term_V"]
    )

    state_variable_dfuns = Final(
        label="Drift functions",
        default={
            "r": "1/tau * ( Delta / (pi * tau) + 2 * V * r)",
            "V": "1/tau * ( V*V - pi*pi*tau*tau*r*r + eta + J * tau * r + I + cr * Coupling_Term_r + cv * Coupling_Term_V)"
        }
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("r", "V"),
        default=("r", "V"),
        doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator.",
    )

    parameter_names = List(
        of=str,
        label="List of parameters for this model",
        default='tau Delta eta J I cr cv'.split())

    state_variables = ('r', 'V')
    _nvar = 2
    # Cvar is the coupling variable. 
    cvar = numpy.array([0, 1], dtype=numpy.int32)
    # Stvar is the variable where stimulus is applied.
    stvar = numpy.array([1], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
            2D model describing the Ott-Antonsen reduction of infinite all-to-all
            coupled QIF neurons (Theta-neurons) as in [Montbrio_Pazo_Roxin_2015]_.

            The two state variables :math:`r` and :math:`V` represent the average
            firing rate and the average membrane potential of our QIF neurons.

            The equations of the infinite QIF 2D population model read

            .. math::
                    \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
                    \dot{V} &= 1/\tau (V^2 - \tau^2 \pi^2 r^2 + \eta + J \tau r + I)
        """

        r, V = state_variables

        # [State_variables, nodes]
        I = self.I
        Delta = self.Delta
        Gamma = self.Gamma
        eta = self.eta
        tau = self.tau
        J = self.J
        cr = self.cr
        cv = self.cv

        Coupling_Term_r = coupling[0, :]  # This zero refers to the first element of cvar (r in this case)
        Coupling_Term_V = coupling[1, :]  # This zero refers to the second element of cvar (V in this case)

        derivative = numpy.empty_like(state_variables)

        derivative[0] = 1 / tau * (Delta / (numpy.pi * tau) + 2 * V * r)
        derivative[1] = 1 / tau * (
                    V ** 2 - numpy.pi ** 2 * tau ** 2 * r ** 2 + eta + J * tau * r + I + cr * Coupling_Term_r + cv * Coupling_Term_V)

        return derivative


class CoombesByrne(Model):
    r"""
    4D model describing the Ott-Antonsen reduction of infinite all-to-all
    coupled QIF neurons (Theta-neurons) as in [Coombes_Byrne_2019]_.
    
    Note: the original equations describe the dynamics of the Kuramoto parameter 
    :math:`Z`. Using the conformal transformation 
    :math:`Z=(1-W^\star)/(1+W^\star)` and :math:`W= \pi r + i V`, 
    we express the system dynamics in terms of two state variables :math:`r` 
    and :math:`V` representing the average firing rate and the average membrane 
    potential of our QIF neurons. The conductance variable and its derivative 
    are :math:`g` and :math:`q`.

    The equations of the model read
    
    .. math::
            \dot{r} &= \Delta/\pi + 2 V r - g r^2 \\
            \dot{V} &= V^2 - \pi^2 r^2 + \eta + (v_{syn} - V) g \\
            \dot{g} &= \alpha q  \\
            \dot{q} &= \alpha (\kappa \pi r - g - 2 q)
            
    .. [Coombes_Byrne_2019] Coombes, S., & Byrne, Á. (2019). Next generation neural mass models. In *Nonlinear Dynamics in Computational Neuroscience* (pp. 1-16). Springer, Cham.
            
    """

    # Define traited attributes for this model, these represent possible kwargs.

    Delta = NArray(
        label=r":math:`\Delta`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width of heterogeneous noise distribution""",
    )

    alpha = NArray(
        label=r":math:`\alpha`",
        default=numpy.array([0.95]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Parameter of the alpha-function""",
    )

    v_syn = NArray(
        label=":math:`v_{syn}`",
        default=numpy.array([-10.0]),
        domain=Range(lo=-20.0, hi=0.0, step=0.01),
        doc="""QIF membrane reversal potential""",
    )

    k = NArray(
        label=":math:`k`",
        default=numpy.array([1.]),
        domain=Range(lo=0., hi=5.0, step=0.01),
        doc="""Local coupling strength""",
    )

    eta = NArray(
        label=r":math:`\eta`",
        default=numpy.array([20.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the
            firing rate variable to itself""",
    )

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "r": numpy.array([0., 6.]),
            "V": numpy.array([-10., 10.]),
            "g": numpy.array([1., 2.]),
            "q": numpy.array([-0.5, 0.7])
        },
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "r": numpy.array([0.0, numpy.inf])
        },
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("r", "V", "g", "q"),
        default=("r", "V"),
        doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator.",
    )

    state_variables = ('r', 'V', 'g', 'q')
    _nvar = 4
    # Cvar is the coupling variable. 
    cvar = numpy.array([0, 1, 2, 3], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
            4D model describing the Ott-Antonsen reduction of infinite all-to-all
            coupled QIF neurons (Theta-neurons) as in [Coombes_Byrne_2019]_.

            The equations of the model read

            .. math::
                    \dot{r} &= \Delta/\pi + 2 V r - g r^2 \\
                    \dot{V} &= V^2 - \pi^2 r^2 + \eta + (v_{syn} - V) g \\
                    \dot{g} &= \alpha q  \\
                    \dot{q} &= \alpha (\kappa \pi r - g - 2 q)
        """
        r, V, g, q = state_variables

        # [State_variables, nodes]
        Delta = self.Delta
        k = self.k
        v_syn = self.v_syn
        eta = self.eta
        alpha = self.alpha

        Coupling_Term_r = coupling[0, :]  # This zero refers to the first element of cvar (r in this case)

        derivative = numpy.empty_like(state_variables)

        derivative[0] = Delta / numpy.pi + 2 * V * r - g * r
        derivative[1] = V ** 2 - numpy.pi ** 2 * r ** 2 + eta + (v_syn - V) * g + Coupling_Term_r
        derivative[2] = alpha * (q)
        derivative[3] = alpha * (k * numpy.pi * r - g - 2 * q)

        return derivative


class CoombesByrne2D(Model):
    r"""
    2D model describing the Ott-Antonsen reduction of infinite all-to-all coupled 
    QIF neurons (Theta-neurons) as in [Coombes_Byrne_2019]_.

    The two state variables :math:`r` and :math:`V` represent the average firing 
    rate and the average membrane potential of our QIF neurons. The conductance 
    :math:`g` is not dynamical and proportional to :math:`r`.

    The equations of the model read
    
    .. math::
            \dot{r} &= \Delta/\pi + 2 V r - g r^2\\
            \dot{V} &= V^2 - \pi^2 r^2 + \eta + (v_{syn} - V) g \\
            g &= \kappa \pi r
    .. [Coombes_Byrne_2019] Coombes, S., & Byrne, Á. (2019). Next generation neural mass models. In *Nonlinear Dynamics in Computational Neuroscience* (pp. 1-16). Springer, Cham.

    """

    # Define traited attributes for this model, these represent possible kwargs.

    Delta = NArray(
        label=r":math:`\Delta`",
        default=numpy.array([1.]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width of heterogeneous noise distribution""",
    )

    v_syn = NArray(
        label=":math:`v_syn`",
        default=numpy.array([-4.0]),
        domain=Range(lo=-20.0, hi=0.0, step=0.01),
        doc="""QIF membrane reversal potential""",
    )

    k = NArray(
        label=":math:`k`",
        default=numpy.array([1.]),
        domain=Range(lo=0., hi=5.0, step=0.01),
        doc="""Local coupling strength""",
    )

    eta = NArray(
        label=r":math:`\eta`",
        default=numpy.array([2.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the
            firing rate variable to itself""",
    )

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"r": numpy.array([0., 2.0]),
                 "V": numpy.array([-2.0, 1.5])},
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "r": numpy.array([0.0, numpy.inf])
        },
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("r", "V"),
        default=("r", "V"),
        doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator.",
    )

    state_variables = ('r', 'V')
    _nvar = 2
    # Cvar is the coupling variable. 
    cvar = numpy.array([0, 1], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
           2D model describing the Ott-Antonsen reduction of infinite all-to-all coupled
           QIF neurons (Theta-neurons) as in [Coombes_Byrne_2019]_.

           The two state variables :math:`r` and :math:`V` represent the average firing
           rate and the average membrane potential of our QIF neurons. The conductance
           :math:`g` is not dynamical and proportional to :math:`r`.

           The equations of the model read

           .. math::
                   \dot{r} &= \Delta/\pi + 2 V r - g r^2\\
                   \dot{V} &= V^2 - \pi^2 r^2 + \eta + (v_{syn} - V) g \\
                   g &= \kappa \pi r
        """
        r, V = state_variables

        # [State_variables, nodes]
        Delta = self.Delta
        k = self.k
        v_syn = self.v_syn
        eta = self.eta

        Coupling_Term_r = coupling[0, :]  # This zero refers to the first element of cvar (r in this case)

        derivative = numpy.empty_like(state_variables)

        derivative[0] = Delta / numpy.pi + 2 * V * r - k * numpy.pi * r ** 2
        derivative[1] = V ** 2 - numpy.pi ** 2 * r ** 2 + eta + (v_syn - V) * k * numpy.pi * r + Coupling_Term_r

        return derivative


class GastSchmidtKnosche_SD(Model):
    r"""
    4D model describing the Ott-Antonsen reduction of infinite all-to-all 
    coupled QIF neurons (Theta-neurons) with Synaptic Depression adaptation 
    mechanisms [Gastetal_2020]_.

    The two state variables :math:`r` and :math:`V` represent the average firing rate and 
    the average membrane potential of our QIF neurons.
    :math:`A` and :math:`B` are respectively the adaptation variable and its derivative.

    The equations of the infinite QIF 2D population model read
    
    .. math::
            \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
            \dot{V} &= 1/\tau (V^2 - \tau^2 \pi^2 r^2 + \eta + J \tau r (1 - A) + I)\\ 
            \dot{A} &= 1/\tau_A (B)\\
            \dot{B} &= 1/\tau_A (-2 B - A + \alpha  r) \\

    .. [Gastetal_2020] Gast, R., Schmidt, H., & Knösche, T. R. (2020). A mean-field description of bursting dynamics in spiking neural networks with short-term adaptation. *Neural Computation*, 32(9), 1615-1634.
    """

    # Define traited attributes for this model, these represent possible kwargs.

    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([1.0]),
        domain=Range(lo=0., hi=15.0, step=0.01),
        doc="""Characteristic time""",
    )

    tau_A = NArray(
        label=r":math:`\tau_A`",
        default=numpy.array([10.0]),
        domain=Range(lo=0., hi=15.0, step=0.01),
        doc="""Adaptation time scale""",
    )

    alpha = NArray(
        label=r":math:`\alpha`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.0, hi=1.0, step=0.1),
        doc="""adaptation rate""",
    )

    I = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""External homogeneous current""",
    )

    Delta = NArray(
        label=r":math:`\Delta`",
        default=numpy.array([2.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width of heterogeneous noise distribution""",
    )

    J = NArray(
        label=":math:`J`",
        default=numpy.array([21.2132]),
        domain=Range(lo=-25.0, hi=25.0, step=0.01),
        doc="""Synaptic weight""",
    )

    eta = NArray(
        label=r":math:`\eta`",
        default=numpy.array([-6.]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Mean of heterogeneous noise distribution""",
    )

    cr = NArray(
        label=":math:`cr`",
        default=numpy.array([1.]),
        domain=Range(lo=0., hi=1, step=0.1),
        doc="""It is the weight on Coupling through variable r.""",
    )

    cv = NArray(
        label=":math:`cv`",
        default=numpy.array([0.]),
        domain=Range(lo=0., hi=1, step=0.1),
        doc="""It is the weight on Coupling through variable V.""",
    )

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"r": numpy.array([0.0, 4]),
                 "V": numpy.array([-3.0, 0.3]),
                 "A": numpy.array([0.0, 0.4]),
                 "B": numpy.array([-0.2, 0.3])},
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "r": numpy.array([0.0, numpy.inf])
        },
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("r", "V", "A", "B"),
        default=("r", "V"),
        doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator.",
    )

    state_variables = ('r', 'V', 'A', 'B')
    _nvar = 4
    # Cvar is the coupling variable. 
    cvar = numpy.array([0, 1, 2, 3], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
            4D model describing the Ott-Antonsen reduction of infinite all-to-all
            coupled QIF neurons (Theta-neurons) with Synaptic Depression adaptation
            mechanisms [Gastetal_2020]_.

            The two state variables :math:`r` and :math:`V` represent the average firing rate and
            the average membrane potential of our QIF neurons.
            :math:`A` and :math:`B` are respectively the adaptation variable and its derivative.

            The equations of the infinite QIF 2D population model read

            .. math::
                    \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
                    \dot{V} &= 1/\tau (V^2 - \tau^2 \pi^2 r^2 + \eta + J \tau r (1 - A) + I)\\
                    \dot{A} &= 1/\tau_A (B)\\
                    \dot{B} &= 1/\tau_A (-2 B - A + \alpha  r) \\
        """
        r, V, A, B = state_variables

        # [State_variables, nodes]
        I = self.I
        Delta = self.Delta
        eta = self.eta
        J = self.J
        alpha = self.alpha
        cr = self.cr
        cv = self.cv
        alpha = self.alpha
        tau_A = self.tau_A
        tau = self.tau

        Coupling_Term_r = coupling[0, :]  # This zero refers to the first element of cvar (r in this case)
        Coupling_Term_V = coupling[1, :]  # This one refers to the second element of cvar (V in this case)

        derivative = numpy.empty_like(state_variables)

        derivative[0] = 1 / tau * (Delta / (numpy.pi * tau) + 2 * V * r)
        derivative[1] = 1 / tau * (V ** 2 - numpy.pi ** 2 * tau ** 2 * r ** 2 + eta + J * tau * r * (
                    1 - A) + I + cr * Coupling_Term_r + cv * Coupling_Term_V)
        derivative[2] = 1 / tau_A * (B)
        derivative[3] = 1 / tau_A * (- 2 * B - A + alpha * r)

        return derivative


class GastSchmidtKnosche_SF(Model):
    r"""
    4D model describing the Ott-Antonsen reduction of infinite all-to-all coupled QIF neurons (Theta-neurons) with Spike-Frequency adaptation mechanisms [Gastetal_2020]_.

    The two state variables :math:`r` and :math:`V` represent the average firing rate and 
    the average membrane potential of our QIF neurons.
    :math:`A` and :math:`B` are respectively the adaptation variable and its derivative.

    The equations of the infinite QIF 2D population model read
    
    .. math::
            \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
            \dot{V} &= 1/\tau (V^2 - \tau^2 \pi^2 r^2 + \eta + J \tau r - A + I)\\ 
            \dot{A} &= 1/\tau_A (B)\\
            \dot{B} &= 1/\tau_A (-2 B - A + \alpha r) \\
    .. [Gastetal_2020]	Gast, R., Schmidt, H., & Knösche, T. R. (2020). A mean-field description of bursting dynamics in spiking neural networks with short-term adaptation. Neural Computation, 32(9), 1615-1634.
    """

    # Define traited attributes for this model, these represent possible kwargs.

    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([1.0]),
        domain=Range(lo=0., hi=15.0, step=0.01),
        doc="""Characteristic time""",
    )

    tau_A = NArray(
        label=r":math:`\tau_A`",
        default=numpy.array([10.0]),
        domain=Range(lo=0., hi=15.0, step=0.01),
        doc="""Adaptation time scale""",
    )

    alpha = NArray(
        label=r":math:`\alpha`",
        default=numpy.array([10.]),
        domain=Range(lo=0.0, hi=1.0, step=0.1),
        doc="""adaptation rate""",
    )

    I = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""External homogeneous current""",
    )

    Delta = NArray(
        label=r":math:`\Delta`",
        default=numpy.array([2.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width of heterogeneous noise distribution""",
    )

    J = NArray(
        label=":math:`J`",
        default=numpy.array([21.2132]),
        domain=Range(lo=-25.0, hi=25.0, step=0.01),
        doc="""Synaptic weight""",
    )

    eta = NArray(
        label=r":math:`\eta`",
        default=numpy.array([1.]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Mean of heterogeneous noise distribution""",
    )

    cr = NArray(
        label=":math:`cr`",
        default=numpy.array([1.]),
        domain=Range(lo=0., hi=1, step=0.1),
        doc="""It is the weight on Coupling through variable r.""",
    )

    cv = NArray(
        label=":math:`cv`",
        default=numpy.array([0.]),
        domain=Range(lo=0., hi=1, step=0.1),
        doc="""It is the weight on Coupling through variable V.""",
    )

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"r": numpy.array([0., 2.0]),
                 "V": numpy.array([-2.0, 1.5]),
                 "A": numpy.array([-1., 1.0]),
                 "B": numpy.array([-1.0, 1.0])},
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "r": numpy.array([0.0, numpy.inf])
        },
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("r", "V", "A", "B"),
        default=("r", "V"),
        doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator.",
    )

    state_variables = ('r', 'V', 'A', 'B')
    _nvar = 4
    # Cvar is the coupling variable. 
    cvar = numpy.array([0, 1, 2, 3], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
            4D model describing the Ott-Antonsen reduction of infinite all-to-all coupled QIF neurons (Theta-neurons) with Spike-Frequency adaptation mechanisms [Gastetal_2020]_.

            The two state variables :math:`r` and :math:`V` represent the average firing rate and
            the average membrane potential of our QIF neurons.
            :math:`A` and :math:`B` are respectively the adaptation variable and its derivative.

            The equations of the infinite QIF 2D population model read

            .. math::
                    \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
                    \dot{V} &= 1/\tau (V^2 - \tau^2 \pi^2 r^2 + \eta + J \tau r - A + I)\\
                    \dot{A} &= 1/\tau_A (B)\\
                    \dot{B} &= 1/\tau_A (-2 B - A + \alpha r) \\
        """
        r, V, A, B = state_variables

        # [State_variables, nodes]
        I = self.I
        Delta = self.Delta
        eta = self.eta
        J = self.J
        alpha = self.alpha
        cr = self.cr
        cv = self.cv
        alpha = self.alpha
        tau_A = self.tau_A
        tau = self.tau

        Coupling_Term_r = coupling[0, :]  # This zero refers to the first element of cvar (r in this case)
        Coupling_Term_V = coupling[1, :]  # This one refers to the second element of cvar (V in this case)

        derivative = numpy.empty_like(state_variables)

        derivative[0] = 1 / tau * (Delta / (numpy.pi * tau) + 2 * V * r)
        derivative[1] = 1 / tau * (
                    V ** 2 - numpy.pi ** 2 * tau ** 2 * r ** 2 + eta + J * tau * r + I - A + cr * Coupling_Term_r + cv * Coupling_Term_V)
        derivative[2] = 1 / tau_A * (B)
        derivative[3] = 1 / tau_A * (- 2 * B - A + alpha * r)

        return derivative


class DumontGutkin(Model):
    r"""
    8D model describing the Ott-Antonsen reduction of infinite all-to-all 
    coupled QIF Excitatory E and Inhibitory I Theta-neurons with local synaptic 
    dynamics [DumontGutkin2019]_.

    State variables :math:`r` and :math:`V` represent the average firing rate and 
    the average membrane potential of our QIF neurons. 
    The neural masses are coupled through the firing rate of :math:`E_i` population from node i-th into :math:`E_j` and :math:`I_j` subpopulations in node j-th.

    The equations of the excitatory infinite QIF 4D population model read (similar for inhibitory):
    
    .. math::
            \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
            \dot{V} &= 1/\tau (V^2 + \eta + \gamma I - \tau^2 \pi^2 r^2 + \tau g - \tau s)\\
            \dot{g} &= 1/\tau_s (-g + J_ r)\\
            \dot{s} &= 1/\tau_s (-s) \\

    .. [DumontGutkin2019] Dumont, G., & Gutkin, B. (2019). Macroscopic phase resetting-curves determine oscillatory coherence and signal transfer in inter-coupled neural circuits. PLoS computational biology, 15(5), e1007019.
    """

    # Define traited attributes for this model, these represent possible kwargs.
    I_e = NArray(
        label=":math:`I_{ext_e}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""External homogeneous current on excitatory population""",
    )

    Delta_e = NArray(
        label=r":math:`\Delta_e`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width of heterogeneous noise distribution over excitatory population""",
    )

    eta_e = NArray(
        label=r":math:`\eta_e`",
        default=numpy.array([-5.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Mean heterogeneous current on excitatory population""",
    )

    tau_e = NArray(
        label=r":math:`\tau_e`",
        default=numpy.array([10.0]),
        domain=Range(lo=0., hi=15.0, step=0.01),
        doc="""Characteristic time of excitatory population""",
    )

    I_i = NArray(
        label=":math:`I_{ext_i}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""External current on inhibitory population""",
    )

    Delta_i = NArray(
        label=r":math:`\Delta_i`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Half-width of heterogeneous noise distribution over inhibitory population""",
    )

    eta_i = NArray(
        label=r":math:`\eta_i`",
        default=numpy.array([-5.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Mean heterogeneous current on inhibitory population""",
    )
    tau_i = NArray(
        label=r":math:`\tau_i`",
        default=numpy.array([10.0]),
        domain=Range(lo=0., hi=15.0, step=0.01),
        doc="""Characteristic time of inhibitory population""",
    )

    tau_s = NArray(
        label=r":math:`\tau_s`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=15.0, step=0.01),
        doc="""Synaptic time constant""",
    )

    J_ee = NArray(
        label=":math:`J_{ee}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-25.0, hi=25.0, step=0.0001),
        doc="""Synaptic weight e-->e""",
    )

    J_ei = NArray(
        label=":math:`J_{ei}`",
        default=numpy.array([10.0]),
        domain=Range(lo=-25.0, hi=25.0, step=0.0001),
        doc="""Synaptic weight i-->e""",
    )

    J_ie = NArray(
        label=":math:`J_{ie}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-25.0, hi=25.0, step=0.0001),
        doc="""Synaptic weight e-->i""",
    )

    J_ii = NArray(
        label=":math:`J_{ii}`",
        default=numpy.array([15.0]),
        domain=Range(lo=-25.0, hi=25.0, step=0.0001),
        doc="""Synaptic weight i-->i""",
    )

    Gamma = NArray(
        label=r":math:`\Gamma`",
        default=numpy.array([5.0]),
        domain=Range(lo=0., hi=10., step=0.1),
        doc="""Ratio of excitatory VS inhibitory global couplings G_ie/G_ee .""",
    )

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"r_e": numpy.array([0., 2.0]),
                 "V_e": numpy.array([-2.0, 1.5]),
                 "s_ee": numpy.array([-1.0, 1.0]),
                 "s_ei": numpy.array([-1.0, 1.0]),
                 "r_i": numpy.array([0., 2.0]),
                 "V_i": numpy.array([-2.0, 1.5]),
                 "s_ie": numpy.array([-1.0, 1.0]),
                 "s_ii": numpy.array([-1.0, 1.0])},
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "r_e": numpy.array([0.0, numpy.inf]),
            "r_i": numpy.array([0.0, numpy.inf])
        },
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("r_e", "V_e", "s_ee", "s_ei", "r_i", "V_i", "s_ie", "s_ii"),
        default=("r_e", "V_e", "s_ee", "s_ei", "r_i", "V_i", "s_ie", "s_ii"),
        doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator.",
    )

    state_variables = ('r_e', 'V_e', 's_ee', 's_ei', 'r_i', 'V_i', 's_ie', 's_ii')
    _nvar = 8
    # Cvar is the coupling variable. 
    cvar = numpy.array([0, 1, 4, 5], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
            8D model describing the Ott-Antonsen reduction of infinite all-to-all
            coupled QIF Excitatory E and Inhibitory I Theta-neurons with local synaptic
            dynamics [DumontGutkin2019]_.

            State variables :math:`r` and :math:`V` represent the average firing rate and
            the average membrane potential of our QIF neurons.
            The neural masses are coupled through the firing rate of :math:`E_i` population from node i-th into :math:`E_j` and :math:`I_j` subpopulations in node j-th.

            The equations of the excitatory infinite QIF 4D population model read (similar for inhibitory):

            .. math::
                    \dot{r} &= 1/\tau (\Delta/(\pi \tau) + 2 V r)\\
                    \dot{V} &= 1/\tau (V^2 + \eta + \gamma I - \tau^2 \pi^2 r^2 + \tau g - \tau s)\\
                    \dot{g} &= 1/\tau_s (-g + J_ r)\\
                    \dot{s} &= 1/\tau_s (-s) \\
        """
        r_e, V_e, s_ee, s_ei, r_i, V_i, s_ie, s_ii = state_variables

        # [State_variables, nodes]
        Delta_e = self.Delta_e
        Delta_i = self.Delta_i
        tau_e = self.tau_e
        tau_i = self.tau_i
        tau_s = self.tau_s
        eta_e = self.eta_e
        eta_i = self.eta_i
        J_ee = self.J_ee
        J_ii = self.J_ii
        J_ei = self.J_ei
        J_ie = self.J_ie
        I_e = self.I_e
        I_i = self.I_i
        Gamma = self.Gamma

        Coupling_Term = coupling[0, :]  # This zero refers to the first element of cvar (r_e in this case)

        derivative = numpy.empty_like(state_variables)

        derivative[0] = 1 / tau_e * (Delta_e / (numpy.pi * tau_e) + 2 * V_e * r_e)
        derivative[1] = 1 / tau_e * (
                    V_e ** 2 + eta_e - tau_e ** 2 * numpy.pi ** 2 * r_e ** 2 + tau_e * s_ee - tau_e * s_ei + I_e)
        derivative[2] = 1 / tau_s * (- s_ee + J_ee * r_e + Coupling_Term)
        derivative[3] = 1 / tau_s * (- s_ei + J_ei * r_i)
        derivative[4] = 1 / tau_i * (Delta_i / (numpy.pi * tau_i) + 2 * V_i * r_i)
        derivative[5] = 1 / tau_i * (
                    V_i ** 2 + eta_i - tau_i ** 2 * numpy.pi ** 2 * r_i ** 2 + tau_i * s_ie - tau_i * s_ii + I_i)
        derivative[6] = 1 / tau_s * (- s_ie + J_ie * r_e + Gamma * Coupling_Term)
        derivative[7] = 1 / tau_s * (- s_ii + J_ii * r_i)

        return derivative




class JansenRit(ModelNumbaDfun):
    r"""
    The Jansen and Rit is a biologically inspired mathematical framework
    originally conceived to simulate the spontaneous electrical activity of
    neuronal assemblies, with a particular focus on alpha activity, for instance,
    as measured by EEG. Later on, it was discovered that in addition to alpha
    activity, this model was also able to simulate evoked potentials.

    .. [JR_1995]  Jansen, B., H. and Rit V., G., *Electroencephalogram and
        visual evoked potential generation in a mathematical model of
        coupled cortical columns*, Biological Cybernetics (73) 357:366, 1995.

    .. [J_1993] Jansen, B., Zouridakis, G. and Brandt, M., *A
        neurophysiologically-based mathematical model of flash visual evoked
        potentials*

    .. figure :: img/JansenRit_45_mode_0_pplane.svg
        :alt: Jansen and Rit phase plane (y4, y5)

        The (:math:`y_4`, :math:`y_5`) phase-plane for the Jansen and Rit model.

    The dynamic equations were taken from [JR_1995]_

    .. math::
        \dot{y_0} &= y_3 \\
        \dot{y_3} &= A a\,S[y_1 - y_2] - 2a\,y_3 - a^2\, y_0 \\
        \dot{y_1} &= y_4\\
        \dot{y_4} &= A a \,[p(t) + \alpha_2 J + S[\alpha_1 J\,y_0]+ c_0]
                    -2a\,y - a^2\,y_1 \\
        \dot{y_2} &= y_5 \\
        \dot{y_5} &= B b (\alpha_4 J\, S[\alpha_3 J \,y_0]) - 2 b\, y_5
                    - b^2\,y_2 \\
        S[v] &= \frac{2\, \nu_{max}}{1 + \exp^{r(v_0 - v)}}

    """

    # Define traited attributes for this model, these represent possible kwargs.
    A = NArray(
        label=":math:`A`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    B = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.05, hi=0.15, step=0.01),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.05]),
        domain=Range(lo=0.025, hi=0.075, step=0.005),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV].

        The usual value for this parameter is 6.0.""")

    nu_max = NArray(
        label=r":math:`\nu_{max}`",
        default=numpy.array([0.0025]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [ms^-1].""")

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")

    J = NArray(
        label=":math:`J`",
        default=numpy.array([135.0]),
        domain=Range(lo=65.0, hi=1350.0, step=1.),
        doc="""Average number of synapses between populations.""")

    a_1 = NArray(
        label=r":math:`\alpha_1`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.""")

    a_2 = NArray(
        label=r":math:`\alpha_2`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop.""")

    a_3 = NArray(
        label=r":math:`\alpha_3`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.""")

    a_4 = NArray(
        label=r":math:`\alpha_4`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.""")

    p_min = NArray(
        label=":math:`p_{min}`",
        default=numpy.array([0.12]),
        domain=Range(lo=0.0, hi=0.12, step=0.01),
        doc="""Minimum input firing rate.""")

    p_max = NArray(
        label=":math:`p_{max}`",
        default=numpy.array([0.32]),
        domain=Range(lo=0.0, hi=0.32, step=0.01),
        doc="""Maximum input firing rate.""")

    mu = NArray(
        label=r":math:`\mu_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"y0": numpy.array([-1.0, 1.0]),
                 "y1": numpy.array([-500.0, 500.0]),
                 "y2": numpy.array([-50.0, 50.0]),
                 "y3": numpy.array([-6.0, 6.0]),
                 "y4": numpy.array([-20.0, 20.0]),
                 "y5": numpy.array([-500.0, 500.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("y0", "y1", "y2", "y3", "y4", "y5"),
        default=("y0", "y1", "y2", "y3"),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""")

    state_variables = tuple('y0 y1 y2 y3 y4 y5'.split())
    _nvar = 6
    cvar = numpy.array([1, 2], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        y0, y1, y2, y3, y4, y5 = state_variables

        # NOTE: This is assumed to be \sum_j u_kj * S[y_{1_j} - y_{2_j}]
        lrc = coupling[0, :]
        short_range_coupling = local_coupling*(y1 - y2)

        # NOTE: for local couplings
        # 0: pyramidal cells
        # 1: excitatory interneurons
        # 2: inhibitory interneurons
        # 0 -> 1,
        # 0 -> 2,
        # 1 -> 0,
        # 2 -> 0,

        exp = numpy.exp
        sigm_y1_y2 = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - (y1 - y2))))
        sigm_y0_1  = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - (self.a_1 * self.J * y0))))
        sigm_y0_3  = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - (self.a_3 * self.J * y0))))

        return numpy.array([
            y3,
            y4,
            y5,
            self.A * self.a * sigm_y1_y2 - 2.0 * self.a * y3 - self.a ** 2 * y0,
            self.A * self.a * (self.mu + self.a_2 * self.J * sigm_y0_1 + lrc + short_range_coupling)
                - 2.0 * self.a * y4 - self.a ** 2 * y1,
            self.B * self.b * (self.a_4 * self.J * sigm_y0_3) - 2.0 * self.b * y5 - self.b ** 2 * y2,
        ])


class ZetterbergJansen(Model):
    """
    Zetterberg et al derived a model inspired by the Wilson-Cowan equations. It served as a basis for the later,
    better known Jansen-Rit model.

    .. [ZL_1978] Zetterberg LH, Kristiansson L and Mossberg K. Performance of a Model for a Local Neuron Population.
        Biological Cybernetics 31, 15-26, 1978.

    .. [JB_1995]  Jansen, B., H. and Rit V., G., *Electroencephalogram and
        visual evoked potential generation in a mathematical model of
        coupled cortical columns*, Biological Cybernetics (73) 357:366, 1995.

    .. [JB_1993] Jansen, B., Zouridakis, G. and Brandt, M., *A
        neurophysiologically-based mathematical model of flash visual evoked
        potentials*

    .. [M_2007] Moran

    .. [S_2010] Spiegler

    .. [A_2012] Auburn

    .. figure :: img/ZetterbergJansen_01_mode_0_pplane.svg
        :alt: Jansen and Rit phase plane

    """

    # Define traited attributes for this model, these represent possible kwargs.
    He = NArray(
        label=":math:`H_e`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    Hi = NArray(
        label=":math:`H_i`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    ke = NArray(
        label=r":math:`\kappa_e`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.05, hi=0.15, step=0.01),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")

    ki = NArray(
        label=r":math:`\kappa_i`",
        default=numpy.array([0.05]),
        domain=Range(lo=0.025, hi=0.075, step=0.005),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")

    e0 = NArray(
        label=r":math:`e_0`",
        default=numpy.array([0.0025]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Half of the maximum population mean firing rate [ms^-1].""")

    rho_2 = NArray(
        label=r":math:`\rho_2`",
        default=numpy.array([6.0]),
        domain=Range(lo=3.12, hi=10.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV]. Population mean firing threshold.""")

    rho_1 = NArray(
        label=r":math:`\rho_1`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")

    gamma_1 = NArray(
        label=r":math:`\gamma_1`",
        default=numpy.array([135.0]),
        domain=Range(lo=65.0, hi=1350.0, step=5.),
        doc="""Average number of synapses between populations (pyramidal to stellate).""")

    gamma_2 = NArray(
        label=r":math:`\gamma_2`",
        default=numpy.array([108.]),
        domain=Range(lo=0.0, hi=200, step=10.0),
        doc="""Average number of synapses between populations (stellate to pyramidal).""")

    gamma_3 = NArray(
        label=r":math:`\gamma_3`",
        default=numpy.array([33.75]),
        domain=Range(lo=0.0, hi=200, step=10.0),
        doc="""Connectivity constant (pyramidal to interneurons)""")

    gamma_4 = NArray(
        label=r":math:`\gamma_4`",
        default=numpy.array([33.75]),
        domain=Range(lo=0.0, hi=200, step=10.0),
        doc="""Connectivity constant (interneurons to pyramidal)""")

    gamma_5 = NArray(
        label=r":math:`\gamma_5`",
        default=numpy.array([15.0]),
        domain=Range(lo=0.0, hi=100, step=10.0),
        doc="""Connectivity constant (interneurons to interneurons)""")

    gamma_1T = NArray(
        label=r":math:`\gamma_{1T}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1000.0, step=5.),
        doc="""Coupling factor from the extrinisic input to the spiny stellate population.""")

    gamma_2T = NArray(
        label=r":math:`\gamma_{2T}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1000.0, step=5.),
        doc="""Coupling factor from the extrinisic input to the pyramidal population.""")

    gamma_3T = NArray(
        label=r":math:`\gamma_{3T}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1000.0, step=5.),
        doc="""Coupling factor from the extrinisic input to the inhibitory population.""")

    P = NArray(
        label=":math:`P`",
        default=numpy.array([0.12]),
        domain=Range(lo=0.0, hi=0.350, step=0.01),
        doc="""Maximum firing rate to the pyramidal population [ms^-1].
        (External stimulus. Constant intensity.Entry point for coupling.)""")

    U = NArray(
        label=":math:`U`",
        default=numpy.array([0.12]),
        domain=Range(lo=0.0, hi=0.350, step=0.01),
        doc="""Maximum firing rate to the stellate population [ms^-1].
        (External stimulus. Constant intensity.Entry point for coupling.)""")

    Q = NArray(
        label=":math:`Q`",
        default=numpy.array([0.12]),
        domain=Range(lo=0.0, hi=0.350, step=0.01),
        doc="""Maximum firing rate to the interneurons population [ms^-1].
        (External stimulus. Constant intensity.Entry point for coupling.)""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"v1": numpy.array([-100.0, 100.0]),
                 "y1": numpy.array([-500.0, 500.0]),
                 "v2": numpy.array([-100.0, 50.0]),
                 "y2": numpy.array([-100.0, 6.0]),
                 "v3": numpy.array([-100.0, 6.0]),
                 "y3": numpy.array([-100.0, 6.0]),
                 "v4": numpy.array([-100.0, 20.0]),
                 "y4": numpy.array([-100.0, 20.0]),
                 "v5": numpy.array([-100.0, 20.0]),
                 "y5": numpy.array([-500.0, 500.0]),
                 "v6": numpy.array([-100.0, 20.0]),
                 "v7": numpy.array([-100.0, 20.0]),},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("v1", "y1", "v2", "y2", "v3", "y3", "v4", "y4", "v5", "y5", "v6", "v7"),
        default=("v6", "v7", "v2", "v3", "v4", "v5"),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The
                                    corresponding state-variable indices for this model are :math:`v_6 = 0`,
                                    :math:`v_7 = 1`, :math:`v_2 = 2`, :math:`v_3 = 3`, :math:`v_4 = 4`, and
                                    :math:`v_5 = 5`""")

    state_variables = tuple('v1 y1 v2 y2 v3 y3 v4 y4 v5 y5 v6 v7'.split())
    _nvar = 12
    cvar = numpy.array([10], dtype=numpy.int32)
    Heke = None  # self.He * self.ke
    Hiki = None  # self.Hi * self.ki
    ke_2 = None  # 2 * self.ke
    ki_2 = None  # 2 * self.ki
    keke = None  # self.ke **2
    kiki = None  # self.ki **2

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """
        Zetterberg et al derived a model inspired by the Wilson-Cowan equations. It served as a basis for the later,
        better known Jansen-Rit model.
        """

        v1 = state_variables[0, :]
        y1 = state_variables[1, :]
        v2 = state_variables[2, :]
        y2 = state_variables[3, :]
        v3 = state_variables[4, :]
        y3 = state_variables[5, :]
        v4 = state_variables[6, :]
        y4 = state_variables[7, :]
        v5 = state_variables[8, :]
        y5 = state_variables[9, :]
        v6 = state_variables[10, :]
        v7 = state_variables[11, :]

        derivative = numpy.empty_like(state_variables)
        # NOTE: long_range_coupling term: coupling variable is v6 . EQUATIONS
        #       ASSUME linear coupling is used. 'coupled_input' represents a rate. It
        #       is very likely that coeffs gamma_xT should be independent for each of the
        #       terms considered as extrinsic input (P, Q, U) (long range coupling) (local coupling)
        #       and noise.

        coupled_input =  self.sigma_fun(coupling[0, :] + local_coupling * v6)

        # exc input to the excitatory interneurons
        derivative[0] = y1
        derivative[1] = self.Heke * (self.gamma_1 * self.sigma_fun(v2 - v3) + self.gamma_1T * (self.U + coupled_input )) - self.ke_2 * y1 - self.keke * v1
        # exc input to the pyramidal cells
        derivative[2] = y2
        derivative[3] = self.Heke * (self.gamma_2 * self.sigma_fun(v1)      + self.gamma_2T * (self.P + coupled_input )) - self.ke_2 * y2 - self.keke * v2
        # inh input to the pyramidal cells
        derivative[4] = y3
        derivative[5] = self.Hiki * (self.gamma_4 * self.sigma_fun(v4 - v5)) - self.ki_2 * y3 - self.kiki * v3
        derivative[6] = y4
        # exc input to the inhibitory interneurons
        derivative[7] = self.Heke * (self.gamma_3 * self.sigma_fun(v2 - v3) + self.gamma_3T * (self.Q + coupled_input)) - self.ke_2 * y4 - self.keke * v4
        derivative[8] = y5
        # inh input to the inhibitory interneurons
        derivative[9] = self.Hiki * (self.gamma_5 * self.sigma_fun(v4 - v5)) - self.ki_2 * y5 - self.keke * v5
        # aux variables (the sum gathering the postsynaptic inh & exc potentials)
        # pyramidal cells
        derivative[10] = y2 - y3
        # inhibitory cells
        derivative[11] = y4 - y5
        return derivative

    def sigma_fun(self, sv):
        """
        Neuronal activation function. This sigmoidal function
        increases from 0 to Q_max as "sv" increases.
        sv represents a membrane potential state variable (V).

        """
        # HACKERY: Hackery for exponential s that blow up.
        # Set to inf, so the result will be effectively zero.
        magic_exp_number = 709
        temp = self.rho_1 * (self.rho_2 - sv)
        temp = numpy.where(temp > magic_exp_number, numpy.inf, temp)
        sigma_v = (2* self.e0) / (1 + numpy.exp(temp))
        return sigma_v

    def update_derived_parameters(self):
        self.Heke = self.He * self.ke
        self.Hiki = self.Hi * self.ki
        self.ke_2 = 2 * self.ke
        self.ki_2 = 2 * self.ki
        self.keke = self.ke**2
        self.kiki = self.ki**2



class LarterBreakspear(Model):
    r"""
    A modified Morris-Lecar model that includes a third equation which simulates
    the effect of a population of inhibitory interneurons synapsing on
    the pyramidal cells.

    .. [Larteretal_1999] Larter et.al. *A coupled ordinary differential equation
        lattice model for the simulation of epileptic seizures.* Chaos. 9(3):
        795, 1999.

    .. [Breaksetal_2003_a] Breakspear, M.; Terry, J. R. & Friston, K. J.  *Modulation of excitatory
        synaptic coupling facilitates synchronization and complex dynamics in an
        onlinear model of neuronal dynamics*. Neurocomputing 52–54 (2003).151–158

    .. [Breaksetal_2003_b] M. J. Breakspear et.al. *Modulation of excitatory
        synaptic coupling facilitates synchronization and complex dynamics in a
        biophysical model of neuronal dynamics.* Network: Computation in Neural
        Systems 14: 703-732, 2003.

    .. [Honeyetal_2007] Honey, C.; Kötter, R.; Breakspear, M. & Sporns, O. * Network structure of
        cerebral cortex shapes functional connectivity on multiple time scales*. (2007)
        PNAS, 104, 10240

    .. [Honeyetal_2009] Honey, C. J.; Sporns, O.; Cammoun, L.; Gigandet, X.; Thiran, J. P.; Meuli,
        R. & Hagmann, P. *Predicting human resting-state functional connectivity
        from structural connectivity.* (2009), PNAS, 106, 2035-2040

    .. [Alstottetal_2009] Alstott, J.; Breakspear, M.; Hagmann, P.; Cammoun, L. & Sporns, O.
        *Modeling the impact of lesions in the human brain*. (2009)),  PLoS Comput Biol, 5, e1000408

    Equations and default parameters are taken from [Breaksetal_2003_b]_.
    All equations and parameters are non-dimensional and normalized.
    For values of d_v  < 0.55, the dynamics of a single column settles onto a
    solitary fixed point attractor.


    Parameters used for simulations in [Breaksetal_2003_a]_ Table 1. Page 153.
    Two nodes were coupled. C=0.1

    +---------------------------+
    |          Table 1          |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | I            |      0.3   |
    +--------------+------------+
    | a_ee         |      0.4   |
    +--------------+------------+
    | a_ei         |      0.1   |
    +--------------+------------+
    | a_ie         |      1.0   |
    +--------------+------------+
    | a_ne         |      1.0   |
    +--------------+------------+
    | a_ni         |      0.4   |
    +--------------+------------+
    | r_NMDA       |      0.2   |
    +--------------+------------+
    | delta        |      0.001 |
    +--------------+------------+
    |   Breakspear et al. 2003  |
    +---------------------------+


    +---------------------------+
    |          Table 2          |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | gK           |      2.0   |
    +--------------+------------+
    | gL           |      0.5   |
    +--------------+------------+
    | gNa          |      6.7   |
    +--------------+------------+
    | gCa          |      1.0   |
    +--------------+------------+
    | a_ne         |      1.0   |
    +--------------+------------+
    | a_ni         |      0.4   |
    +--------------+------------+
    | a_ee         |      0.36  |
    +--------------+------------+
    | a_ei         |      2.0   |
    +--------------+------------+
    | a_ie         |      2.0   |
    +--------------+------------+
    | VK           |     -0.7   |
    +--------------+------------+
    | VL           |     -0.5   |
    +--------------+------------+
    | VNa          |      0.53  |
    +--------------+------------+
    | VCa          |      1.0   |
    +--------------+------------+
    | phi          |      0.7   |
    +--------------+------------+
    | b            |      0.1   |
    +--------------+------------+
    | I            |      0.3   |
    +--------------+------------+
    | r_NMDA       |      0.25  |
    +--------------+------------+
    | C            |      0.1   |
    +--------------+------------+
    | TCa          |     -0.01  |
    +--------------+------------+
    | d_Ca         |      0.15  |
    +--------------+------------+
    | TK           |      0.0   |
    +--------------+------------+
    | d_K          |      0.3   |
    +--------------+------------+
    | VT           |      0.0   |
    +--------------+------------+
    | ZT           |      0.0   |
    +--------------+------------+
    | TNa          |      0.3   |
    +--------------+------------+
    | d_Na         |      0.15  |
    +--------------+------------+
    | d_V          |      0.65  |
    +--------------+------------+
    | d_Z          |      d_V   |
    +--------------+------------+
    | QV_max       |      1.0   |
    +--------------+------------+
    | QZ_max       |      1.0   |
    +--------------+------------+
    |   Alstott et al. 2009     |
    +---------------------------+


    NOTES about parameters

    :math:`\delta_V` : for :math:`\delta_V` < 0.55, in an uncoupled network,
    the system exhibits fixed point dynamics; for 0.55 < :math:`\delta_V` < 0.59,
    limit cycle attractors; and for :math:`\delta_V` > 0.59 chaotic attractors
    (eg, d_V=0.6,aee=0.5,aie=0.5, gNa=0, Iext=0.165)

    :math:`\delta_Z`
    this parameter might be spatialized: ones(N,1).*0.65 + modn*(rand(N,1)-0.5);

    :math:`C`
    The long-range coupling :math:`\delta_C` is ‘weak’ in the sense that
    the model is well behaved for parameter values for which C < a_ee and C << a_ie.



    .. figure :: img/LarterBreakspear_01_mode_0_pplane.svg
            :alt: Larter-Breaskpear phase plane (V, W)

            The (:math:`V`, :math:`W`) phase-plane for the Larter-Breakspear model.


    Dynamic equations:

    .. math::
            \dot{V}_k & = - (g_{Ca} + (1 - C) \, r_{NMDA} \, a_{ee} \, Q_V + C \, r_{NMDA} \, a_{ee} \, \langle Q_V\rangle^{k}) \, m_{Ca} \, (V - VCa) \\
                           & \,\,- g_K \, W \, (V - VK) -  g_L \, (V - VL) \\
                           & \,\,- (g_{Na} \, m_{Na} + (1 - C) \, a_{ee} \, Q_V + C \, a_{ee} \, \langle Q_V\rangle^{k}) \,(V - VNa) \\
                           & \,\,- a_{ie} \, Z \, Q_Z + a_{ne} \, I \\
                           & \\
            \dot{W}_k & = \phi \, \dfrac{m_K - W}{\tau_{K}} \\
                           & \nonumber\\
            \dot{Z}_k &= b (a_{ni}\, I + a_{ei}\,V\,Q_V) \\
            Q_{V}   &= Q_{V_{max}} \, (1 + \tanh\left(\dfrac{V_{k} - VT}{\delta_{V}}\right)) \\
            Q_{Z}   &= Q_{Z_{max}} \, (1 + \tanh\left(\dfrac{Z_{k} - ZT}{\delta_{Z}}\right))

    See Equations (7), (3), (6) and (2) respectively in [Breaksetal_2003_a]_.
    Pag: 705-706

    """

    # Define traited attributes for this model, these represent possible kwargs.
    gCa = NArray(
        label=":math:`g_{Ca}`",
        default=numpy.array([1.1]),
        domain=Range(lo=0.9, hi=1.5, step=0.1),
        doc="""Conductance of population of Ca++ channels.""")

    gK = NArray(
        label=":math:`g_{K}`",
        default=numpy.array([2.0]),
        domain=Range(lo=1.95, hi= 2.05, step=0.025),
        doc="""Conductance of population of K channels.""")

    gL = NArray(
        label=":math:`g_{L}`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.45 , hi=0.55, step=0.05),
        doc="""Conductance of population of leak channels.""")

    phi = NArray(
        label=r":math:`\phi`",
        default=numpy.array([0.7]),
        domain=Range(lo=0.3, hi=0.9, step=0.1),
        doc="""Temperature scaling factor.""")

    gNa = NArray(
        label=":math:`g_{Na}`",
        default=numpy.array([6.7]),
        domain=Range(lo=0.0, hi=10.0, step=0.1),
        doc="""Conductance of population of Na channels.""")

    TK = NArray(
        label=":math:`T_{K}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=0.0001, step=0.00001),
        doc="""Threshold value for K channels.""")

    TCa = NArray(
        label=":math:`T_{Ca}`",
        default=numpy.array([-0.01]),
        domain=Range(lo=-0.02, hi=-0.01, step=0.0025),
        doc="Threshold value for Ca channels.")

    TNa = NArray(
        label=":math:`T_{Na}`",
        default=numpy.array([0.3]),
        domain=Range(lo=0.25, hi= 0.3, step=0.025),
        doc="Threshold value for Na channels.")

    VCa = NArray(
        label=":math:`V_{Ca}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.9, hi=1.1, step=0.05),
        doc="""Ca Nernst potential.""")

    VK = NArray(
        label=":math:`V_{K}`",
        default=numpy.array([-0.7]),
        domain=Range(lo=-0.8, hi=1., step=0.1),
        doc="""K Nernst potential.""")

    VL = NArray(
        label=":math:`V_{L}`",
        default=numpy.array([-0.5]),
        domain=Range(lo=-0.7, hi=-0.4, step=0.1),
        doc="""Nernst potential leak channels.""")

    VNa = NArray(
        label=":math:`V_{Na}`",
        default=numpy.array([0.53]),
        domain=Range(lo=0.51, hi=0.55, step=0.01),
        doc="""Na Nernst potential.""")

    d_K = NArray(
        label=r":math:`\delta_{K}`",
        default=numpy.array([0.3]),
        domain=Range(lo=0.1, hi=0.4, step=0.1),
        doc="""Variance of K channel threshold.""")

    tau_K = NArray(
        label=r":math:`\tau_{K}`",
        default=numpy.array([1.0]),
        domain=Range(lo=1.0, hi=10.0, step=1.0),
        doc="""Time constant for K relaxation time (ms)""")

    d_Na = NArray(
        label=r":math:`\delta_{Na}`",
        default=numpy.array([0.15]),
        domain=Range(lo=0.1, hi=0.2, step=0.05),
        doc="Variance of Na channel threshold.")

    d_Ca = NArray(
        label=r":math:`\delta_{Ca}`",
        default=numpy.array([0.15]),
        domain=Range(lo=0.1, hi=0.2, step=0.05),
        doc="Variance of Ca channel threshold.")

    aei = NArray(
        label=":math:`a_{ei}`",
        default=numpy.array([2.0]),
        domain=Range(lo=0.1, hi=2.0, step=0.1),
        doc="""Excitatory-to-inhibitory synaptic strength.""")

    aie = NArray(
        label=":math:`a_{ie}`",
        default=numpy.array([2.0]),
        domain=Range(lo=0.5, hi=2.0, step=0.1),
        doc="""Inhibitory-to-excitatory synaptic strength.""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.0001, hi=1.0, step=0.0001),
        doc="""Time constant scaling factor. The original value is 0.1""")

    C = NArray(
        label=":math:`C`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Strength of excitatory coupling. Balance between internal and
        local (and global) coupling strength. C > 0 introduces interdependences between
        consecutive columns/nodes. C=1 corresponds to maximum coupling between node and no self-coupling.
        This strenght should be set to sensible values when a whole network is connected. """)

    ane = NArray(
        label=":math:`a_{ne}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.4, hi=1.0, step=0.05),
        doc="""Non-specific-to-excitatory synaptic strength.""")

    ani = NArray(
        label=":math:`a_{ni}`",
        default=numpy.array([0.4]),
        domain=Range(lo=0.3, hi=0.5, step=0.05),
        doc="""Non-specific-to-inhibitory synaptic strength.""")

    aee = NArray(
        label=":math:`a_{ee}`",
        default=numpy.array([0.4]),
        domain=Range(lo=0.0, hi=0.6, step=0.05),
        doc="""Excitatory-to-excitatory synaptic strength.""")

    Iext = NArray(
       label=":math:`I_{ext}`",
       default=numpy.array([0.3]),
       domain=Range(lo=0.165, hi=0.3, step=0.005),
       doc="""Subcortical input strength. It represents a non-specific
       excitation or thalamic inputs.""")

    rNMDA = NArray(
        label=":math:`r_{NMDA}`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.2, hi=0.3, step=0.05),
        doc="""Ratio of NMDA to AMPA receptors.""")

    VT = NArray(
        label=":math:`V_{T}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=0.7, step=0.01),
        doc="""Threshold potential (mean) for excitatory neurons.
        In [Breaksetal_2003_b]_ this value is 0.""")

    d_V = NArray(
        label=r":math:`\delta_{V}`",
        default=numpy.array([0.65]),
        domain=Range(lo=0.49, hi=0.7, step=0.01),
        doc="""Variance of the excitatory threshold. It is one of the main
        parameters explored in [Breaksetal_2003_b]_.""")

    ZT = NArray(
        label=":math:`Z_{T}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=0.1, step=0.005),
        doc="""Threshold potential (mean) for inihibtory neurons.""")

    d_Z = NArray(
        label=r":math:`\delta_{Z}`",
        default=numpy.array([0.7]),
        domain=Range(lo=0.001, hi=0.75, step=0.05),
        doc="""Variance of the inhibitory threshold.""")

    # NOTE: the values were not in the article.
    QV_max = NArray(
        label=":math:`QV_{max}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.1, hi=1., step=0.001),
        doc="""Maximal firing rate for excitatory populations (kHz)""")

    QZ_max = NArray(
        label=":math:`QZ_{max}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.1, hi=1., step=0.001),
        doc="""Maximal firing rate for excitatory populations (kHz)""")

    t_scale = NArray(
        label=":math:`t_{scale}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.1, hi=1., step=0.001),
        doc="""Time scale factor""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("V", "W", "Z"),
        default=("V",),
        doc="""This represents the default state-variables of this Model to be
        monitored. It can be overridden for each Monitor if desired.""")

    # Informational attribute, used for phase-plane and initial()
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "V": numpy.array([-1.5, 1.5]),
            "W": numpy.array([-1.5, 1.5]),
            "Z": numpy.array([-1.5, 1.5])},
        doc="""The values for each state-variable should be set to encompass
            the expected dynamic range of that state-variable for the current
            parameters, it is used as a mechanism for bounding random inital
            conditions when the simulation isn't started from an explicit
            history, it is also provides the default range of phase-plane plots.""")

    state_variables = tuple('V W Z'.split())
    _state_variables = ("V", "W", "Z")
    _nvar = 3
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        Dynamic equations:

        .. math::
            \dot{V}_k & = - (g_{Ca} + (1 - C) \, r_{NMDA} \, a_{ee} \, Q_V + C \, r_{NMDA} \, a_{ee} \, \langle Q_V\rangle^{k}) \, m_{Ca} \, (V - VCa) \\
                           & \,\,- g_K \, W \, (V - VK) -  g_L \, (V - VL) \\
                           & \,\,- (g_{Na} \, m_{Na} + (1 - C) \, a_{ee} \, Q_V + C \, a_{ee} \, \langle Q_V\rangle^{k}) \,(V - VNa) \\
                           & \,\,- a_{ie} \, Z \, Q_Z + a_{ne} \, I \\
                           & \\
            \dot{W}_k & = \phi \, \dfrac{m_K - W}{\tau_{K}} \\
                           & \nonumber\\
            \dot{Z}_k &= b (a_{ni}\, I + a_{ei}\,V\,Q_V) \\
            Q_{V}   &= Q_{V_{max}} \, (1 + \tanh\left(\dfrac{V_{k} - VT}{\delta_{V}}\right)) \\
            Q_{Z}   &= Q_{Z_{max}} \, (1 + \tanh\left(\dfrac{Z_{k} - ZT}{\delta_{Z}}\right))

        """
        V, W, Z = state_variables
        derivative = numpy.empty_like(state_variables)
        c_0   = coupling[0, :]
        # relationship between membrane voltage and channel conductance
        m_Ca = 0.5 * (1 + numpy.tanh((V - self.TCa) / self.d_Ca))
        m_Na = 0.5 * (1 + numpy.tanh((V - self.TNa) / self.d_Na))
        m_K  = 0.5 * (1 + numpy.tanh((V - self.TK )  / self.d_K))
        # voltage to firing rate
        QV    = 0.5 * self.QV_max * (1 + numpy.tanh((V - self.VT) / self.d_V))
        QZ    = 0.5 * self.QZ_max * (1 + numpy.tanh((Z - self.ZT) / self.d_Z))
        lc_0  = local_coupling * QV
        derivative[0] = self.t_scale * (- (self.gCa + (1.0 - self.C) * (self.rNMDA * self.aee) * (QV + lc_0)+ self.C * self.rNMDA * self.aee * c_0) * m_Ca * (V - self.VCa)
                         - self.gK * W * (V - self.VK)
                         - self.gL * (V - self.VL)
                         - (self.gNa * m_Na + (1.0 - self.C) * self.aee * (QV  + lc_0) + self.C * self.aee * c_0) * (V - self.VNa)
                         - self.aie * Z * QZ
                         + self.ane * self.Iext)
        derivative[1] = self.t_scale * self.phi * (m_K - W) / self.tau_K
        derivative[2] = self.t_scale * self.b * (self.ani * self.Iext + self.aei * V * QV)
        return derivative



class Linear(Model):
    gamma = NArray(
        label=r":math:`\gamma`",
        default=numpy.array([-10.0]),
        domain=Range(lo=-100.0, hi=0.0, step=1.0),
        doc="The damping coefficient specifies how quickly the node's activity relaxes, must be larger"
            " than the node's in-degree in order to remain stable.")

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"x": numpy.array([-1, 1])},
        doc="Range used for state variable initialization and visualization.")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("x",),
        default=("x",), )

    coupling_terms = Final(
        label="Coupling terms",
        # how to unpack coupling array
        default=["c"]
    )

    state_variable_dfuns = Final(
        label="Drift functions",
        default={
            "x": "gamma * x + c",
        }
    )

    parameter_names = List(
        of=str,
        label="List of parameters for this model",
        default=tuple('gamma'.split()))

    state_variables = ('x',)
    _nvar = 1
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, state, coupling, local_coupling=0.0):
        r"""
        .. math::
            x = a{\gamma} + b
        """
        x, = state
        c, = coupling
        dx = self.gamma * x + c + local_coupling * x
        return numpy.array([dx])


class Generic2dOscillator(ModelNumbaDfun):
    r"""
    The Generic2dOscillator model is a generic dynamic system with two state
    variables. The dynamic equations of this model are composed of two ordinary
    differential equations comprising two nullclines. The first nullcline is a
    cubic function as it is found in most neuron and population models; the
    second nullcline is arbitrarily configurable as a polynomial function up to
    second order. The manipulation of the latter nullcline's parameters allows
    to generate a wide range of different behaviours.

    Equations:

    .. math::
            \dot{V} &= d \, \tau (-f V^3 + e V^2 + g V + \alpha W + \gamma I) \\
            \dot{W} &= \dfrac{d}{\tau}\,\,(c V^2 + b V - \beta W + a)

    See:


        .. [FH_1961] FitzHugh, R., *Impulses and physiological states in theoretical
            models of nerve membrane*, Biophysical Journal 1: 445, 1961.

        .. [Nagumo_1962] Nagumo et.al, *An Active Pulse Transmission Line Simulating
            Nerve Axon*, Proceedings of the IRE 50: 2061, 1962.

        .. [SJ_2011] Stefanescu, R., Jirsa, V.K. *Reduced representations of
            heterogeneous mixed neural networks with synaptic coupling*.
            Physical Review E, 83, 2011.

        .. [SJ_2010]	Jirsa VK, Stefanescu R.  *Neural population modes capture
            biologically realistic large-scale network dynamics*. Bulletin of
            Mathematical Biology, 2010.

        .. [SJ_2008_a] Stefanescu, R., Jirsa, V.K. *A low dimensional description
            of globally coupled heterogeneous neural networks of excitatory and
            inhibitory neurons*. PLoS Computational Biology, 4(11), 2008).


    The model's (:math:`V`, :math:`W`) time series and phase-plane its nullclines
    can be seen in the figure below.

    The model with its default parameters exhibits FitzHugh-Nagumo like dynamics.

    +---------------------------+
    |  Table 1                  |
    +--------------+------------+
    |  EXCITABLE CONFIGURATION  |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | a            |     -2.0   |
    +--------------+------------+
    | b            |    -10.0   |
    +--------------+------------+
    | c            |      0.0   |
    +--------------+------------+
    | d            |      0.02  |
    +--------------+------------+
    | I            |      0.0   |
    +--------------+------------+
    |  limit cycle if a is 2.0  |
    +---------------------------+


    +---------------------------+
    |   Table 2                 |
    +--------------+------------+
    |   BISTABLE CONFIGURATION  |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | a            |      1.0   |
    +--------------+------------+
    | b            |      0.0   |
    +--------------+------------+
    | c            |     -5.0   |
    +--------------+------------+
    | d            |      0.02  |
    +--------------+------------+
    | I            |      0.0   |
    +--------------+------------+
    | monostable regime:        |
    | fixed point if Iext=-2.0  |
    | limit cycle if Iext=-1.0  |
    +---------------------------+


    +---------------------------+
    |  Table 3                  |
    +--------------+------------+
    |  EXCITABLE CONFIGURATION  |
    +--------------+------------+
    |  (similar to Morris-Lecar)|
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | a            |      0.5   |
    +--------------+------------+
    | b            |      0.6   |
    +--------------+------------+
    | c            |     -4.0   |
    +--------------+------------+
    | d            |      0.02  |
    +--------------+------------+
    | I            |      0.0   |
    +--------------+------------+
    | excitable regime if b=0.6 |
    | oscillatory if b=0.4      |
    +---------------------------+


    +---------------------------+
    |  Table 4                  |
    +--------------+------------+
    |  GhoshetAl,  2008         |
    |  KnocketAl,  2009         |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | a            |    1.05    |
    +--------------+------------+
    | b            |   -1.00    |
    +--------------+------------+
    | c            |    0.0     |
    +--------------+------------+
    | d            |    0.1     |
    +--------------+------------+
    | I            |    0.0     |
    +--------------+------------+
    | alpha        |    1.0     |
    +--------------+------------+
    | beta         |    0.2     |
    +--------------+------------+
    | gamma        |    -1.0    |
    +--------------+------------+
    | e            |    0.0     |
    +--------------+------------+
    | g            |    1.0     |
    +--------------+------------+
    | f            |    1/3     |
    +--------------+------------+
    | tau          |    1.25    |
    +--------------+------------+
    |                           |
    |  frequency peak at 10Hz   |
    |                           |
    +---------------------------+


    +---------------------------+
    |  Table 5                  |
    +--------------+------------+
    |  SanzLeonetAl  2013       |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | a            |    - 0.5   |
    +--------------+------------+
    | b            |    -10.0   |
    +--------------+------------+
    | c            |      0.0   |
    +--------------+------------+
    | d            |      0.02  |
    +--------------+------------+
    | I            |      0.0   |
    +--------------+------------+
    |                           |
    |  intrinsic frequency is   |
    |  approx 10 Hz             |
    |                           |
    +---------------------------+

    NOTE: This regime, if I = 2.1, is called subthreshold regime.
    Unstable oscillations appear through a subcritical Hopf bifurcation.


    .. figure :: img/Generic2dOscillator_01_mode_0_pplane.svg
    .. _phase-plane-Generic2D:
        :alt: Phase plane of the generic 2D population model with (V, W)

        The (:math:`V`, :math:`W`) phase-plane for the generic 2D population
        model for default parameters. The dynamical system has an equilibrium
        point.

    .. automethod:: Generic2dOscillator.dfun

    """

    # Define traited attributes for this model, these represent possible kwargs.
    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([1.0]),
        domain=Range(lo=1.0, hi=5.0, step=0.01),
        doc="""A time-scale hierarchy can be introduced for the state
        variables :math:`V` and :math:`W`. Default parameter is 1, which means
        no time-scale hierarchy.""")

    I = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.01),
        doc="""Baseline shift of the cubic nullcline""")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([-2.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.01),
        doc="""Vertical shift of the configurable nullcline""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([-10.0]),
        domain=Range(lo=-20.0, hi=15.0, step=0.01),
        doc="""Linear slope of the configurable nullcline""")

    c = NArray(
        label=":math:`c`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""Parabolic term of the configurable nullcline""")

    d = NArray(
        label=":math:`d`",
        default=numpy.array([0.02]),
        domain=Range(lo=0.0001, hi=1.0, step=0.0001),
        doc="""Temporal scale factor. Warning: do not use it unless
        you know what you are doing and know about time tides.""")

    e = NArray(
        label=":math:`e`",
        default=numpy.array([3.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Coefficient of the quadratic term of the cubic nullcline.""")

    f = NArray(
        label=":math:`f`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Coefficient of the cubic term of the cubic nullcline.""")

    g = NArray(
        label=":math:`g`",
        default=numpy.array([0.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.5),
        doc="""Coefficient of the linear term of the cubic nullcline.""")

    alpha = NArray(
        label=r":math:`\alpha`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the
            slow variable to the fast variable.""")

    beta = NArray(
        label=r":math:`\beta`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the
            slow variable to itself""")

    # This parameter is basically a hack to avoid having a negative lower boundary in the global coupling strength.
    gamma = NArray(
        label=r":math:`\gamma`",
        default=numpy.array([1.0]),
        domain=Range(lo=-1.0, hi=1.0, step=0.1),
        doc="""Constant parameter to reproduce FHN dynamics where
               excitatory input currents are negative.
               It scales both I and the long range coupling term.""")

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"V": numpy.array([-2.0, 4.0]),
                 "W": numpy.array([-6.0, 6.0])},
        doc="""The values for each state-variable should be set to encompass
            the expected dynamic range of that state-variable for the current
            parameters, it is used as a mechanism for bounding random initial
            conditions when the simulation isn't started from an explicit
            history, it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("V", "W", "V + W", "V - W"),
        default=("V",),
        doc="The quantities of interest for monitoring for the generic 2D oscillator.")

    state_variables = ('V', 'W')
    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        V = state_variables[0, :]
        W = state_variables[1, :]

        # [State_variables, nodes]
        c_0 = coupling[0, :]

        tau = self.tau
        I = self.I
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        e = self.e
        f = self.f
        g = self.g
        beta = self.beta
        alpha = self.alpha
        gamma = self.gamma

        lc_0 = local_coupling * V

        # Pre-allocate the result array then instruct numexpr to use it as output.
        # This avoids an expensive array concatenation
        derivative = numpy.empty_like(state_variables)

        # ev = RefBase.evaluate
        # ev('d * tau * (alpha * W - f * V**3 + e * V**2 + g * V + gamma * I + gamma *c_0 + lc_0)', out=derivative[0])
        # ev('d * (a + b * V + c * V**2 - beta * W) / tau', out=derivative[1])
        # we don't have numexpr in browser, so we just use numpy
        derivative[0] = d * tau * (alpha * W - f * V**3 + e * V**2 + g * V + gamma * I + gamma *c_0 + lc_0)
        derivative[1] = d * (a + b * V + c * V**2 - beta * W) / tau

        return derivative


class Kuramoto(Model):
    r"""
    The Kuramoto model is a model of synchronization phenomena derived by
    Yoshiki Kuramoto in 1975 which has since been applied to diverse domains
    including the study of neuronal oscillations and synchronization.

    See:

        .. [YK_1975] Y. Kuramoto, in: H. Arakai (Ed.), International Symposium
            on Mathematical Problems in Theoretical Physics, *Lecture Notes in
            Physics*, page 420, vol. 39, 1975.

        .. [SS_2000] S. H. Strogatz. *From Kuramoto to Crawford: exploring the
            onset of synchronization in populations of coupled oscillators*.
            Physica D, 143, 2000.

        .. [JC_2011] J. Cabral, E. Hugues, O. Sporns, G. Deco. *Role of local
            network oscillations in resting-state functional connectivity*.
            NeuroImage, 57, 1, 2011.

    The :math:`\theta` variable is the phase angle of the oscillation.

    Dynamic equations:
        .. math::

                \dot{\theta}_{k} = \omega_{k} + \mathbf{\Gamma}(\theta_k, \theta_j, u_{kj}) + \sin(W_{\zeta}\theta)

    """

    # Define traited attributes for this model, these represent possible kwargs.
    omega = NArray(
        label=r":math:`\omega`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.01, hi=200.0, step=0.1),
        doc=r""":math:`\omega` sets the base line frequency for the
            Kuramoto oscillator in [rad/ms]""")

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"theta": numpy.array([0.0, numpy.pi * 2.0]), },
        doc="""The values for each state-variable should be set to encompass
            the expected dynamic range of that state-variable for the current
            parameters, it is used as a mechanism for bounding random initial
            conditions when the simulation isn't started from an explicit
            history, it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("theta",),
        default=("theta",),
        doc="""This represents the default state-variables of this Model to be
                            monitored. It can be overridden for each Monitor if desired. The Kuramoto
                            model, however, only has one state variable with and index of 0, so it
                            is not necessary to change the default here.""")

    state_variables = ['theta']
    _nvar = 1
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The :math:`\theta` variable is the phase angle of the oscillation.

        .. math::
            \dot{\theta}_{k} = \omega_{k} + \mathbf{\Gamma}(\theta_k, \theta_j, u_{kj}) + \sin(W_{\zeta}\theta)

        where :math:`I` is the input via local and long range connectivity,
        passing first through the Kuramoto coupling function,

        """

        theta = state_variables[0, :]
        # import pdb; pdb.set_trace()

        # A) Distribution of phases according to the local connectivity kernel
        local_range_coupling = numpy.sin(local_coupling * theta)

        # NOTE: To evaluate.
        # B) Strength of the interactions
        # local_range_coupling = local_coupling * numpy.sin(theta)

        I = coupling[0, :] + local_range_coupling

        if not hasattr(self, 'derivative'):
            self.derivative = numpy.empty((1,) + theta.shape)

        # phase update
        self.derivative[0] = self.omega + I

        # all this pi makeh me have great hungary, can has sum NaN?
        return self.derivative


class SupHopf(ModelNumbaDfun):
    r"""
    The supHopf model describes the normal form of a supercritical Hopf bifurcation in Cartesian coordinates.
    This normal form has a supercritical bifurcation at a=0 with a the bifurcation parameter in the model. So 
    for a < 0, the local dynamics has a stable fixed point and the system corresponds to a damped oscillatory 
    state, whereas for a > 0, the local dynamics enters in a stable limit cycle and the system switches to an 
    oscillatory state.

    See for examples:

    .. [Kuznetsov_2013] Kuznetsov, Y.A. *Elements of applied bifurcation theory.* Springer Sci & Business
        Media, 2013, vol. 112.

    .. [Deco_2017a] Deco, G., Kringelbach, M.L., Jirsa, V.K., Ritter, P. *The dynamics of resting fluctuations
       in the brain: metastability and its dynamical cortical core* Sci Reports, 2017, 7: 3095.

    The equations of the supHopf equations read as follows:

    .. math::
        \dot{x}_{i} &= (a_{i} - x_{i}^{2} - y_{i}^{2})x_{i} - {\omega}{i}y_{i} \\
        \dot{y}_{i} &= (a_{i} - x_{i}^{2} - y_{i}^{2})y_{i} + {\omega}{i}x_{i}

    where a is the local bifurcation parameter and omega the angular frequency.
    """

    a = NArray(
        label=r":math:`a`",
        default=numpy.array([-0.5]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""Local bifurcation parameter.""")

    omega = NArray(
        label=r":math:`\omega`",
        default=numpy.array([1.]),
        domain=Range(lo=0.05, hi=630.0, step=0.01),
        doc="""Angular frequency.""")

    # Initialization.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"x": numpy.array([-5.0, 5.0]),
                 "y": numpy.array([-5.0, 5.0])},
        doc="""The values for each state-variable should be set to encompass
               the expected dynamic range of that state-variable for the current
               parameters, it is used as a mechanism for bounding random initial
               conditions when the simulation isn't started from an explicit
               history, it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("x", "y"),
        default=("x",),
        doc="Quantities of supHopf available to monitor.")

    state_variables = ["x", "y"]

    _nvar = 2  # number of state-variables
    cvar = numpy.array([0, 1], dtype=numpy.int32)  # coupling variables

    def dfun(self, state_variables, coupling, local_coupling=0.0,
                    array=numpy.array, where=numpy.where, concat=numpy.concatenate):
        y = state_variables
        ydot = numpy.empty_like(state_variables)

        # long-range coupling
        c_0 = coupling[0]
        c_1 = coupling[1]

        # short-range (local) coupling
        lc_0 = local_coupling * y[0]

        # supHopf's equations in Cartesian coordinates:
        ydot[0] = (self.a - y[0] ** 2 - y[1] ** 2) * y[0] - self.omega * y[1] + c_0 + lc_0
        ydot[1] = (self.a - y[0] ** 2 - y[1] ** 2) * y[1] + self.omega * y[0] + c_1

        return ydot



class WilsonCowan(ModelNumbaDfun):
    r"""
    **References**:

    .. [WC_1972] Wilson, H.R. and Cowan, J.D. *Excitatory and inhibitory
        interactions in localized populations of model neurons*, Biophysical
        journal, 12: 1-24, 1972.
    .. [WC_1973] Wilson, H.R. and Cowan, J.D  *A Mathematical Theory of the
        Functional Dynamics of Cortical and Thalamic Nervous Tissue*

    .. [D_2011] Daffertshofer, A. and van Wijk, B. *On the influence of
        amplitude on the connectivity between phases*
        Frontiers in Neuroinformatics, July, 2011

    Used Eqns 11 and 12 from [WC_1972]_ in ``dfun``.  P and Q represent external
    inputs, which when exploring the phase portrait of the local model are set
    to constant values. However in the case of a full network, P and Q are the
    entry point to our long range and local couplings, that is, the  activity
    from all other nodes is the external input to the local population.

    The default parameters are taken from figure 4 of [WC_1972]_, pag. 10

    +---------------------------+
    |          Table 0          |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | k_e, k_i     |    0.00    |
    +--------------+------------+
    | r_e, r_i     |    0.00    |
    +--------------+------------+
    | tau_e, tau_i |    9.0     |
    +--------------+------------+
    | c_ee         |    11.0    |
    +--------------+------------+
    | c_ei         |    3.0     |
    +--------------+------------+
    | c_ie         |    12.0    |
    +--------------+------------+
    | c_ii         |    10.0    |
    +--------------+------------+
    | a_e          |    0.2     |
    +--------------+------------+
    | a_i          |    0.0     |
    +--------------+------------+
    | b_e          |    1.8     |
    +--------------+------------+
    | b_i          |    3.0     |
    +--------------+------------+
    | theta_e      |    -1.0    |
    +--------------+------------+
    | theta_i      |    -1.0    |
    +--------------+------------+
    | alpha_e      |    1.0     |
    +--------------+------------+
    | alpha_i      |    1.0     |
    +--------------+------------+
    | P            |    -1.0    |
    +--------------+------------+
    | Q            |    -1.0    |
    +--------------+------------+
    | c_e, c_i     |    0.0     |
    +--------------+------------+
    | shift_sigmoid|    True    |
    +--------------+------------+

    In [WC_1973]_ they present a model of neural tissue on the pial surface is.
    See Fig. 1 in page 58. The following local couplings (lateral interactions)
    occur given a region i and a region j:

      E_i-> E_j
      E_i-> I_j
      I_i-> I_j
      I_i-> E_j


    +---------------------------+
    |          Table 1          |
    +--------------+------------+
    |                           |
    |  SanzLeonetAl,   2014     |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | k_e, k_i     |    1.00    |
    +--------------+------------+
    | r_e, r_i     |    0.00    |
    +--------------+------------+
    | tau_e, tau_i |    10.0    |
    +--------------+------------+
    | c_ee         |    10.0    |
    +--------------+------------+
    | c_ei         |    6.0     |
    +--------------+------------+
    | c_ie         |    10.0    |
    +--------------+------------+
    | c_ii         |    1.0     |
    +--------------+------------+
    | a_e, a_i     |    1.0     |
    +--------------+------------+
    | b_e, b_i     |    0.0     |
    +--------------+------------+
    | theta_e      |    2.0     |
    +--------------+------------+
    | theta_i      |    3.5     |
    +--------------+------------+
    | alpha_e      |    1.2     |
    +--------------+------------+
    | alpha_i      |    2.0     |
    +--------------+------------+
    | P            |    0.5     |
    +--------------+------------+
    | Q            |    0.0     |
    +--------------+------------+
    | c_e, c_i     |    1.0     |
    +--------------+------------+
    | shift_sigmoid|    False   |
    +--------------+------------+
    |                           |
    |  frequency peak at 20  Hz |
    |                           |
    +---------------------------+


    The parameters in Table 1 reproduce Figure A1 in  [D_2011]_
    but set the limit cycle frequency to a sensible value (eg, 20Hz).

    Model bifurcation parameters:
        * :math:`c_1`
        * :math:`P`



    The models (:math:`E`, :math:`I`) phase-plane, including a representation of
    the vector field as well as its nullclines, using default parameters, can be
    seen below:

        .. _phase-plane-WC:
        .. figure :: img/WilsonCowan_01_mode_0_pplane.svg
            :alt: Wilson-Cowan phase plane (E, I)

            The (:math:`E`, :math:`I`) phase-plane for the Wilson-Cowan model.


    The general formulation for the \textit{\textbf{Wilson-Cowan}} model as a
    dynamical unit at a node $k$ in a BNM with $l$ nodes reads:

    .. math::
            \dot{E}_k &= \dfrac{1}{\tau_e} (-E_k  + (k_e - r_e E_k) \mathcal{S}_e (\alpha_e \left( c_{ee} E_k - c_{ei} I_k  + P_k - \theta_e + \mathbf{\Gamma}(E_k, E_j, u_{kj}) + W_{\zeta}\cdot E_j + W_{\zeta}\cdot I_j\right) ))\\
            \dot{I}_k &= \dfrac{1}{\tau_i} (-I_k  + (k_i - r_i I_k) \mathcal{S}_i (\alpha_i \left( c_{ie} E_k - c_{ee} I_k  + Q_k - \theta_i + \mathbf{\Gamma}(E_k, E_j, u_{kj}) + W_{\zeta}\cdot E_j + W_{\zeta}\cdot I_j\right) ))

    """

    # Define traited attributes for this model, these represent possible kwargs.
    c_ee = NArray(
        label=":math:`c_{ee}`",
        default=numpy.array([12.0]),
        domain=Range(lo=11.0, hi=16.0, step=0.01),
        doc="""Excitatory to excitatory  coupling coefficient""")

    c_ei = NArray(
        label=":math:`c_{ei}`",
        default=numpy.array([4.0]),
        domain=Range(lo=2.0, hi=15.0, step=0.01),
        doc="""Inhibitory to excitatory coupling coefficient""")

    c_ie = NArray(
        label=":math:`c_{ie}`",
        default=numpy.array([13.0]),
        domain=Range(lo=2.0, hi=22.0, step=0.01),
        doc="""Excitatory to inhibitory coupling coefficient.""")

    c_ii = NArray(
        label=":math:`c_{ii}`",
        default=numpy.array([11.0]),
        domain=Range(lo=2.0, hi=15.0, step=0.01),
        doc="""Inhibitory to inhibitory coupling coefficient.""")

    tau_e = NArray(
        label=r":math:`\tau_e`",
        default=numpy.array([10.0]),
        domain=Range(lo=0.0, hi=150.0, step=0.01),
        doc="""Excitatory population, membrane time-constant [ms]""")

    tau_i = NArray(
        label=r":math:`\tau_i`",
        default=numpy.array([10.0]),
        domain=Range(lo=0.0, hi=150.0, step=0.01),
        doc="""Inhibitory population, membrane time-constant [ms]""")

    a_e = NArray(
        label=":math:`a_e`",
        default=numpy.array([1.2]),
        domain=Range(lo=0.0, hi=1.4, step=0.01),
        doc="""The slope parameter for the excitatory response function""")

    b_e = NArray(
        label=":math:`b_e`",
        default=numpy.array([2.8]),
        domain=Range(lo=1.4, hi=6.0, step=0.01),
        doc="""Position of the maximum slope of the excitatory sigmoid function""")

    c_e = NArray(
        label=":math:`c_e`",
        default=numpy.array([1.0]),
        domain=Range(lo=1.0, hi=20.0, step=1.0),
        doc="""The amplitude parameter for the excitatory response function""")

    theta_e = NArray(
        label=r":math:`\theta_e`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=60., step=0.01),
        doc="""Excitatory threshold""")

    a_i = NArray(
        label=":math:`a_i`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""The slope parameter for the inhibitory response function""")

    b_i = NArray(
        label=":math:`b_i`",
        default=numpy.array([4.0]),
        domain=Range(lo=2.0, hi=6.0, step=0.01),
        doc="""Position of the maximum slope of a sigmoid function [in
        threshold units]""")

    theta_i = NArray(
        label=r":math:`\theta_i`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=60.0, step=0.01),
        doc="""Inhibitory threshold""")

    c_i = NArray(
        label=":math:`c_i`",
        default=numpy.array([1.0]),
        domain=Range(lo=1.0, hi=20.0, step=1.0),
        doc="""The amplitude parameter for the inhibitory response function""")

    r_e = NArray(
        label=":math:`r_e`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=2.0, step=0.01),
        doc="""Excitatory refractory period""")

    r_i = NArray(
        label=":math:`r_i`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=2.0, step=0.01),
        doc="""Inhibitory refractory period""")

    k_e = NArray(
        label=":math:`k_e`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=2.0, step=0.01),
        doc="""Maximum value of the excitatory response function""")

    k_i = NArray(
        label=":math:`k_i`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Maximum value of the inhibitory response function""")

    P = NArray(
        label=":math:`P`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=20.0, step=0.01),
        doc="""External stimulus to the excitatory population.
        Constant intensity.Entry point for coupling.""")

    Q = NArray(
        label=":math:`Q`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=20.0, step=0.01),
        doc="""External stimulus to the inhibitory population.
        Constant intensity.Entry point for coupling.""")

    alpha_e = NArray(
        label=r":math:`\alpha_e`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=20.0, step=0.01),
        doc="""External stimulus to the excitatory population.
        Constant intensity.Entry point for coupling.""")

    alpha_i = NArray(
        label=r":math:`\alpha_i`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=20.0, step=0.01),
        doc="""External stimulus to the inhibitory population.
        Constant intensity.Entry point for coupling.""")

    shift_sigmoid = NArray(
        dtype= numpy.bool_,
        label=r":math:`shift sigmoid`",
        default=numpy.array([True]),
        doc="""In order to have resting state (E=0 and I=0) in absence of external input,
        the logistic curve are translated downward S(0)=0""",
        )

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"E": numpy.array([0.0, 1.0]),
                 "I": numpy.array([0.0, 1.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("E", "I", "E + I", "E - I"),
        default=("E",),
        doc="""This represents the default state-variables of this Model to be
               monitored. It can be overridden for each Monitor if desired. The
               corresponding state-variable indices for this model are :math:`E = 0`
               and :math:`I = 1`.""")

    state_variables = 'E I'.split()
    _nvar = 2
    cvar = numpy.array([0, 1], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""

        .. math::
            \tau \dot{x}(t) &= -z(t) + \phi(z(t)) \\
            \phi(x) &= \frac{c}{1-exp(-a (x-b))}

        """

        E = state_variables[0, :]
        I = state_variables[1, :]
        derivative = numpy.empty_like(state_variables)

        # long-range coupling
        c_0 = coupling[0, :]

        # short-range (local) coupling
        lc_0 = local_coupling * E
        lc_1 = local_coupling * I

        x_e = self.alpha_e * (self.c_ee * E - self.c_ei * I + self.P  - self.theta_e +  c_0 + lc_0 + lc_1)
        x_i = self.alpha_i * (self.c_ie * E - self.c_ii * I + self.Q  - self.theta_i + lc_0 + lc_1)

        if self.shift_sigmoid:
            s_e = self.c_e * (1.0 / (1.0 + numpy.exp(-self.a_e * (x_e - self.b_e))) - 1.0
                              / (1.0 + numpy.exp(-self.a_e * -self.b_e)))
            s_i = self.c_i * (1.0 / (1.0 + numpy.exp(-self.a_i * (x_i - self.b_i))) - 1.0
                              / (1.0 + numpy.exp(-self.a_i * -self.b_i)))
        else:
            s_e = self.c_e / (1.0 + numpy.exp(-self.a_e * (x_e - self.b_e)))
            s_i = self.c_i / (1.0 + numpy.exp(-self.a_i * (x_i - self.b_i)))

        derivative[0] = (-E + (self.k_e - self.r_e * E) * s_e) / self.tau_e
        derivative[1] = (-I + (self.k_i - self.r_i * I) * s_i) / self.tau_i

        return derivative


class ReducedWongWangExcInh(ModelNumbaDfun):
    r"""
    .. [WW_2006] Kong-Fatt Wong and Xiao-Jing Wang,  *A Recurrent Network
                Mechanism of Time Integration in Perceptual Decisions*.
                Journal of Neuroscience 26(4), 1314-1328, 2006.

    .. [DPA_2014] Deco Gustavo, Ponce Alvarez Adrian, Patric Hagmann,
                  Gian Luca Romani, Dante Mantini, and Maurizio Corbetta. *How Local
                  Excitation–Inhibition Ratio Impacts the Whole Brain Dynamics*.
                  The Journal of Neuroscience 34(23), 7886 –7898, 2014.


    Equations taken from [DPA_2013]_ , page 11242

    .. math::
                 x_{ek}       &=   w_p\,J_N \, S_{ek} - J_iS_{ik} + W_eI_o + GJ_N \mathbf\Gamma(S_{ek}, S_{ej}, u_{kj}) \\
                 H(x_{ek})    &=  \dfrac{a_ex_{ek}- b_e}{1 - \exp(-d_e(a_ex_{ek} -b_e))} \\
                 \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}) \, {\gamma}H(x_{ek}) \\

                 x_{ik}       &=   J_N \, S_{ek} - S_{ik} + W_iI_o + {\lambda}GJ_N \mathbf\Gamma(S_{ik}, S_{ej}, u_{kj}) \\
                 H(x_{ik})    &=  \dfrac{a_ix_{ik} - b_i}{1 - \exp(-d_i(a_ix_{ik} -b_i))} \\
                 \dot{S}_{ik} &= -\dfrac{S_{ik}}{\tau_i} + \gamma_iH(x_{ik}) \

    """

    # Define traited attributes for this model, these represent possible kwargs.

    a_e = NArray(
        label=":math:`a_e`",
        default=numpy.array([310., ]),
        domain=Range(lo=0., hi=500., step=1.),
        doc="[n/C]. Excitatory population input gain parameter, chosen to fit numerical solutions.")

    b_e = NArray(
        label=":math:`b_e`",
        default=numpy.array([125., ]),
        domain=Range(lo=0., hi=200., step=1.),
        doc="[Hz]. Excitatory population input shift parameter chosen to fit numerical solutions.")

    d_e = NArray(
        label=":math:`d_e`",
        default=numpy.array([0.160, ]),
        domain=Range(lo=0.0, hi=0.2, step=0.001),
        doc="""[s]. Excitatory population input scaling parameter chosen to fit numerical solutions.""")

    gamma_e = NArray(
        label=r":math:`\gamma_e`",
        default=numpy.array([0.641/1000, ]),
        domain=Range(lo=0.0, hi=1.0/1000, step=0.01/1000),
        doc="""Excitatory population kinetic parameter""")

    tau_e = NArray(
        label=r":math:`\tau_e`",
        default=numpy.array([100., ]),
        domain=Range(lo=50., hi=150., step=1.),
        doc="""[ms]. Excitatory population NMDA decay time constant.""")

    w_p = NArray(
        label=r":math:`w_p`",
        default=numpy.array([1.4, ]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Excitatory population recurrence weight""")

    J_N = NArray(
        label=r":math:`J_N`",
        default=numpy.array([0.15, ]),
        domain=Range(lo=0.001, hi=0.5, step=0.001),
        doc="""[nA] NMDA current""")

    W_e = NArray(
        label=r":math:`W_e`",
        default=numpy.array([1.0, ]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Excitatory population external input scaling weight""")

    a_i = NArray(
        label=":math:`a_i`",
        default=numpy.array([615., ]),
        domain=Range(lo=0., hi=1000., step=1.),
        doc="[n/C]. Inhibitory population input gain parameter, chosen to fit numerical solutions.")

    b_i = NArray(
        label=":math:`b_i`",
        default=numpy.array([177.0, ]),
        domain=Range(lo=0.0, hi=200.0, step=1.0),
        doc="[Hz]. Inhibitory population input shift parameter chosen to fit numerical solutions.")

    d_i = NArray(
        label=":math:`d_i`",
        default=numpy.array([0.087, ]),
        domain=Range(lo=0.0, hi=0.2, step=0.001),
        doc="""[s]. Inhibitory population input scaling parameter chosen to fit numerical solutions.""")

    gamma_i = NArray(
        label=r":math:`\gamma_i`",
        default=numpy.array([1.0/1000, ]),
        domain=Range(lo=0.0, hi=2.0/1000, step=0.01/1000),
        doc="""Inhibitory population kinetic parameter""")

    tau_i = NArray(
        label=r":math:`\tau_i`",
        default=numpy.array([10., ]),
        domain=Range(lo=5., hi=100., step=1.0),
        doc="""[ms]. Inhibitory population NMDA decay time constant.""")

    J_i = NArray(
        label=r":math:`J_{i}`",
        default=numpy.array([1.0, ]),
        domain=Range(lo=0.001, hi=2.0, step=0.001),
        doc="""[nA] Local inhibitory current""")

    W_i = NArray(
        label=r":math:`W_i`",
        default=numpy.array([0.7, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Inhibitory population external input scaling weight""")

    I_o = NArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.382, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.001),
        doc="""[nA]. Effective external input""")

    I_ext = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.001),
        doc="""[nA]. Effective external stimulus input""")

    G = NArray(
        label=":math:`G`",
        default=numpy.array([2.0, ]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Global coupling scaling""")

    lamda = NArray(
        label=r":math:`\lambda`",
        default=numpy.array([0.0, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Inhibitory global coupling scaling""")

    state_variable_range = Final(
        default={
            "S_e": numpy.array([0.0, 1.0]),
            "S_i": numpy.array([0.0, 1.0])
        },
        label="State variable ranges [lo, hi]",
        doc="Population firing rate")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"S_e": numpy.array([0.0, 1.0]), "S_i": numpy.array([0.0, 1.0])},
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. Set None for one-sided boundaries""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('S_e', 'S_i'),
        default=('S_e', 'S_i'),
        doc="""default state variables to be monitored""")

    state_variables = ['S_e', 'S_i']
    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)


    def dfun(self, state_variables, coupling, local_coupling=0.0):
        S = state_variables[:, :]

        c_0 = coupling[0, :]

        # if applicable
        lc_0 = local_coupling * S[0]

        coupling = self.G * self.J_N * (c_0 + lc_0)

        J_N_S_e = self.J_N * S[0]

        x_e = self.w_p * J_N_S_e - self.J_i * S[1] + self.W_e * self.I_o + coupling + self.I_ext

        x_e = self.a_e * x_e - self.b_e
        H_e = x_e / (1 - numpy.exp(-self.d_e * x_e))

        dS_e = - (S[0] / self.tau_e) + (1 - S[0]) * H_e * self.gamma_e

        x_i = J_N_S_e - S[1] + self.W_i * self.I_o + self.lamda * coupling

        x_i = self.a_i * x_i - self.b_i
        H_i = x_i / (1 - numpy.exp(-self.d_i * x_i))

        dS_i = - (S[1] / self.tau_i) + H_i * self.gamma_i

        derivative = numpy.array([dS_e, dS_i])

        return derivative


class DecoBalancedExcInh(ReducedWongWangExcInh):
    r"""
    .. [Deco_2021] Deco, Gustavo, Morten L. Kringelbach, Aurina Arnatkeviciute,
    Stuart Oldham, Kristina Sabaroedin, Nigel C. Rogasch, Kevin M. Aquino, and
    Alex Fornito. "Dynamical consequences of regional heterogeneity in the
    brain’s transcriptional landscape." Science Advances 7, no. 29 (2021):
    eabf4752.

    Equations extend the [DPA_2013] with effective gain parameter M_i to


    .. math::
                 x_{ek}       &=   w_p\,J_N \, S_{ek} - J_iS_{ik} + W_eI_o + GJ_N \mathbf\Gamma(S_{ek}, S_{ej}, u_{kj}) \\
                 H(x_{ek})    &=  \dfrac{M_i(a_ex_{ek}- b_e)}{1 - \exp(-d_e M_i(a_ex_{ek} -b_e))} \\
                 \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}) \, {\gamma}H(x_{ek}) \\

                 x_{ik}       &=   J_N \, S_{ek} - S_{ik} + W_iI_o + {\lambda}GJ_N \mathbf\Gamma(S_{ik}, S_{ej}, u_{kj}) \\
                 H(x_{ik})    &=  \dfrac{M_i(a_ix_{ik} - b_i)}{1 - \exp(-d_i M_i(a_ix_{ik} -b_i))} \\
                 \dot{S}_{ik} &= -\dfrac{S_{ik}}{\tau_i} + \gamma_iH(x_{ik}) \
    """

    # Define traited attributes for this model, these represent possible kwargs.

    M_i = NArray(
        label=":math:`ratio`",
        default=numpy.array([1.0, ]),
        domain=Range(lo=1.0, hi=10., step=0.01),
        doc="""Effective gain within a region.""")


    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        Numpy dfun for transcriptional model presented in [Deco_2020],
        Dynamical consequences of regional heterogeneity in the brain’s
        transcriptional landscape
        """

        S = state_variables[:, :]

        S_e = S[0, :]
        S_i = S[1, :]

        c_0 = coupling[0, :]

        # if applicable
        lc_0 = local_coupling * S_e

        coupling = self.G * self.J_N * (c_0 + lc_0)

        J_N_S_e = self.J_N * S_e

        inh = self.J_i * S_i

        I_e = self.W_e * self.I_o + self.w_p * J_N_S_e + coupling - inh + self.I_ext

        x_e = (self.a_e * I_e - self.b_e) * self.M_i
        H_e = x_e / (1 - numpy.exp(-self.d_e * x_e))

        dS_e = - (S_e / self.tau_e) + (1.0 - S_e) * H_e * self.gamma_e

        I_i = self.W_i * self.I_o + J_N_S_e - S_i + self.lamda * coupling

        x_i = (self.a_i * I_i - self.b_i) * self.M_i
        H_i = x_i / (1 - numpy.exp(-self.d_i * x_i))

        dS_i = - (S_i / self.tau_i) + H_i * self.gamma_i

        derivative = numpy.array([dS_e, dS_i])

        return derivative


class ReducedWongWang(ModelNumbaDfun):
    r"""
    .. [WW_2006] Kong-Fatt Wong and Xiao-Jing Wang,  *A Recurrent Network
                Mechanism of Time Integration in Perceptual Decisions*.
                Journal of Neuroscience 26(4), 1314-1328, 2006.

    .. [DPA_2013] Deco Gustavo, Ponce Alvarez Adrian, Dante Mantini, Gian Luca
                  Romani, Patric Hagmann and Maurizio Corbetta. *Resting-State
                  Functional Connectivity Emerges from Structurally and
                  Dynamically Shaped Slow Linear Fluctuations*. The Journal of
                  Neuroscience 32(27), 11239-11252, 2013.


    Equations taken from [DPA_2013]_ , page 11242

    .. math::
                 x_k       &=   w\,J_N \, S_k + I_o + J_N \mathbf\Gamma(S_k, S_j, u_{kj})\\
                 H(x_k)    &=  \dfrac{ax_k - b}{1 - \exp(-d(ax_k -b))}\\
                 \dot{S}_k &= -\dfrac{S_k}{\tau_s} + (1 - S_k) \, H(x_k) \, \gamma

    """

    # Define traited attributes for this model, these represent possible kwargs.
    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.270, ]),
        domain=Range(lo=0.0, hi=0.270, step=0.01),
        doc="[n/C]. Input gain parameter, chosen to fit numerical solutions.")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.108, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="[kHz]. Input shift parameter chosen to fit numerical solutions.")

    d = NArray(
        label=":math:`d`",
        default=numpy.array([154., ]),
        domain=Range(lo=0.0, hi=200.0, step=0.01),
        doc="""[ms]. Parameter chosen to fit numerical solutions.""")

    gamma = NArray(
        label=r":math:`\gamma`",
        default=numpy.array([0.641, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Kinetic parameter""")

    tau_s = NArray(
        label=r":math:`\tau_S`",
        default=numpy.array([100., ]),
        domain=Range(lo=50.0, hi=150.0, step=1.0),
        doc="""Kinetic parameter. NMDA decay time constant.""")

    w = NArray(
        label=r":math:`w`",
        default=numpy.array([0.6, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Excitatory recurrence""")

    J_N = NArray(
        label=r":math:`J_{N}`",
        default=numpy.array([0.2609, ]),
        domain=Range(lo=0.2609, hi=0.5, step=0.001),
        doc="""Excitatory recurrence""")

    I_o = NArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.33, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""[nA] Effective external input""")

    sigma_noise = NArray(
        label=r":math:`\sigma_{noise}`",
        default=numpy.array([0.000000001, ]),
        domain=Range(lo=0.0, hi=0.005, step=0.0001),
        doc="""[nA] Noise amplitude. Take this value into account for stochatic
        integration schemes.""")

    state_variable_range = Final(
        label="State variable ranges [lo, hi]",
        default={"S": numpy.array([0.0, 1.0])},
        doc="Population firing rate")

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"S": numpy.array([0.0, 1.0])},
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. Set None for one-sided boundaries""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("S",),
        default=("S",),
        doc="""default state variables to be monitored""")

    state_variables = ['S']
    _nvar = 1
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        S = state_variables[0, :]

        c_0 = coupling[0, :]


        # if applicable
        lc_0 = local_coupling * S

        x  = self.w * self.J_N * S + self.I_o + self.J_N * c_0 + self.J_N * lc_0
        H = (self.a * x - self.b) / (1 - numpy.exp(-self.d * (self.a * x - self.b)))
        dS = - (S / self.tau_s) + (1 - S) * H * self.gamma

        derivative = numpy.array([dS])
        return derivative


class ReducedSetBase(Model):
    number_of_modes = 3
    nu = 1500
    nv = 1500


class ReducedSetFitzHughNagumo(ReducedSetBase):
    r"""
    A reduced representation of a set of Fitz-Hugh Nagumo oscillators,
    [SJ_2008]_.

    The models (:math:`\xi`, :math:`\eta`) phase-plane, including a
    representation of the vector field as well as its nullclines, using default
    parameters, can be seen below:

        .. _phase-plane-rFHN_0:
        .. figure :: img/ReducedSetFitzHughNagumo_01_mode_0_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 1st mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the first mode of
            a reduced set of Fitz-Hugh Nagumo oscillators.

        .. _phase-plane-rFHN_1:
        .. figure :: img/ReducedSetFitzHughNagumo_01_mode_1_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 2nd mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the second mode of
            a reduced set of Fitz-Hugh Nagumo oscillators.

        .. _phase-plane-rFHN_2:
        .. figure :: img/ReducedSetFitzHughNagumo_01_mode_2_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 3rd mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the third mode of
            a reduced set of Fitz-Hugh Nagumo oscillators.


    The system's equations for the i-th mode at node q are:

    .. math::
                \dot{\xi}_{i}    &=  c\left(\xi_i-e_i\frac{\xi_{i}^3}{3} -\eta_{i}\right)
                                  + K_{11}\left[\sum_{k=1}^{o} A_{ik}\xi_k-\xi_i\right]
                                  - K_{12}\left[\sum_{k =1}^{o} B_{i k}\alpha_k-\xi_i\right] + cIE_i \\
                                 &\, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                  +  \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr} \right] \\
                \dot{\eta}_i     &= \frac{1}{c}\left(\xi_i-b\eta_i+m_i\right) \\
                & \\
                \dot{\alpha}_i   &= c\left(\alpha_i-f_i\frac{\alpha_i^3}{3}-\beta_i\right)
                                  + K_{21}\left[\sum_{k=1}^{o} C_{ik}\xi_i-\alpha_i\right] + cII_i \\
                                 & \, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                  + \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr}\right] \\
                                 & \\
                \dot{\beta}_i    &= \frac{1}{c}\left(\alpha_i-b\beta_i+n_i\right)

    .. automethod:: ReducedSetFitzHughNagumo.update_derived_parameters

    #NOTE: In the Article this modelis called StefanescuJirsa2D

    """

    # Define traited attributes for this model, these represent possible kwargs.
    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([3.0]),
        domain=Range(lo=1.5, hi=4.5, step=0.01),
        doc="""doc...(prob something about timescale seperation)""")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.45]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""doc...""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.9]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""doc...""")

    K11 = NArray(
        label=":math:`K_{11}`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, excitatory to excitatory""")

    K12 = NArray(
        label=":math:`K_{12}`",
        default=numpy.array([0.15]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, inhibitory to excitatory""")

    K21 = NArray(
        label=":math:`K_{21}`",
        default=numpy.array([0.15]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, excitatory to inhibitory""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.35]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Standard deviation of Gaussian distribution""")

    mu = NArray(
        label=r":math:`\mu`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Mean of Gaussian distribution""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"xi": numpy.array([-4.0, 4.0]),
                 "eta": numpy.array([-3.0, 3.0]),
                 "alpha": numpy.array([-4.0, 4.0]),
                 "beta": numpy.array([-3.0, 3.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("xi", "eta", "alpha", "beta"),
        default=("xi", "alpha"),
        doc=r"""This represents the default state-variables of this Model to be
                monitored. It can be overridden for each Monitor if desired. The
                corresponding state-variable indices for this model are :math:`\xi = 0`,
                :math:`\eta = 1`, :math:`\alpha = 2`, and :math:`\beta= 3`.""")

    state_variables = tuple('xi eta alpha beta'.split())
    _nvar = 4
    cvar = numpy.array([0, 2], dtype=numpy.int32)
    # Derived parameters
    Aik = None
    Bik = None
    Cik = None
    e_i = None
    f_i = None
    IE_i = None
    II_i = None
    m_i = None
    n_i = None

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""


        The system's equations for the i-th mode at node q are:

        .. math::
                \dot{\xi}_{i}    &=  c\left(\xi_i-e_i\frac{\xi_{i}^3}{3} -\eta_{i}\right)
                                  + K_{11}\left[\sum_{k=1}^{o} A_{ik}\xi_k-\xi_i\right]
                                  - K_{12}\left[\sum_{k =1}^{o} B_{i k}\alpha_k-\xi_i\right] + cIE_i                       \\
                                 &\, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                  +  \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr} \right] \\
                \dot{\eta}_i     &= \frac{1}{c}\left(\xi_i-b\eta_i+m_i\right)                                              \\
                & \\
                \dot{\alpha}_i   &= c\left(\alpha_i-f_i\frac{\alpha_i^3}{3}-\beta_i\right)
                                  + K_{21}\left[\sum_{k=1}^{o} C_{ik}\xi_i-\alpha_i\right] + cII_i                          \\
                                 & \, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                  + \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr}\right] \\
                                 & \\
                \dot{\beta}_i    &= \frac{1}{c}\left(\alpha_i-b\beta_i+n_i\right)

        """

        xi = state_variables[0, :]
        eta = state_variables[1, :]
        alpha = state_variables[2, :]
        beta = state_variables[3, :]
        derivative = numpy.empty_like(state_variables)
        # sum the activity from the modes
        c_0 = coupling[0, :].sum(axis=1)[:, numpy.newaxis]

        # TODO: generalize coupling variables to a matrix form
        # c_1 = coupling[1, :] # this cv represents alpha

        # for batching, we convert dot to einsum, so
        # dot(xi, Aik) results in an error of
        # ValueError: shapes (32,3,8) and (3,3,1) not aligned
        # instead use an einsum
        # numpy.einsum('nmb,mMo->nMb', xi, self.Aik)

        derivative[0] = (self.tau * (xi - self.e_i * xi ** 3 / 3.0 - eta) +
               self.K11 * (numpy.einsum('nmb,mMo->nMb', xi, self.Aik) - xi) -
               self.K12 * (numpy.einsum('nmb,mMo->nMb', alpha, self.Bik) - xi) +
               self.tau * (self.IE_i + c_0 + local_coupling * xi))

        derivative[1] = (xi - self.b * eta + self.m_i) / self.tau

        derivative[2] = (self.tau * (alpha - self.f_i * alpha ** 3 / 3.0 - beta) +
                  self.K21 * (numpy.einsum('nmb,mMo->nMb', xi, self.Cik) - alpha) +
                  self.tau * (self.II_i + c_0 + local_coupling * xi))

        derivative[3] = (alpha - self.b * beta + self.n_i) / self.tau

        return derivative

    def update_derived_parameters(self):
        """
        Calculate coefficients for the Reduced FitzHugh-Nagumo oscillator based
        neural field model. Specifically, this method implements equations for
        calculating coefficients found in the supplemental material of
        [SJ_2008]_.

        Include equations here...

        """
        newaxis = numpy.newaxis
        from scipy.integrate import trapezoid as trapz
        from scipy.stats import norm as scipy_stats_norm

        stepu = 1.0 / (self.nu + 2 - 1)
        stepv = 1.0 / (self.nv + 2 - 1)

        norm = scipy_stats_norm(loc=self.mu, scale=self.sigma)

        Zu = norm.ppf(numpy.arange(stepu, 1.0, stepu))
        Zv = norm.ppf(numpy.arange(stepv, 1.0, stepv))

        # Define the modes
        V = numpy.zeros((self.number_of_modes, self.nv))
        U = numpy.zeros((self.number_of_modes, self.nu))

        nv_per_mode = self.nv // self.number_of_modes
        nu_per_mode = self.nu // self.number_of_modes

        for i in range(self.number_of_modes):
            V[i, i * nv_per_mode:(i + 1) * nv_per_mode] = numpy.ones(nv_per_mode)
            U[i, i * nu_per_mode:(i + 1) * nu_per_mode] = numpy.ones(nu_per_mode)

        # Normalise the modes
        V = V / numpy.tile(numpy.sqrt(trapz(V * V, Zv, axis=1)), (self.nv, 1)).T
        U = U / numpy.tile(numpy.sqrt(trapz(U * U, Zu, axis=1)), (self.nv, 1)).T

        # Get Normal PDF's evaluated with sampling Zv and Zu
        g1 = norm.pdf(Zv)
        g2 = norm.pdf(Zu)
        G1 = numpy.tile(g1, (self.number_of_modes, 1))
        G2 = numpy.tile(g2, (self.number_of_modes, 1))

        cV = numpy.conj(V)
        cU = numpy.conj(U)

        intcVdZ = trapz(cV, Zv, axis=1)[:, newaxis]
        intG1VdZ = trapz(G1 * V, Zv, axis=1)[newaxis, :]
        intcUdZ = trapz(cU, Zu, axis=1)[:, newaxis]
        # import pdb; pdb.set_trace()
        # Calculate coefficients
        self.Aik = numpy.dot(intcVdZ, intG1VdZ).T
        self.Bik = numpy.dot(intcVdZ, trapz(G2 * U, Zu, axis=1)[newaxis, :])
        self.Cik = numpy.dot(intcUdZ, intG1VdZ).T

        self.e_i = trapz(cV * V ** 3, Zv, axis=1)[newaxis, :]
        self.f_i = trapz(cU * U ** 3, Zu, axis=1)[newaxis, :]

        self.IE_i = trapz(Zv * cV, Zv, axis=1)[newaxis, :]
        self.II_i = trapz(Zu * cU, Zu, axis=1)[newaxis, :]

        self.m_i = (self.a * intcVdZ).T
        self.n_i = (self.a * intcUdZ).T
        # import pdb; pdb.set_trace()

        # batching requires to add a last axis of size 1
        self.Aik = self.Aik[..., newaxis]
        self.Bik = self.Bik[..., newaxis]
        self.Cik = self.Cik[..., newaxis]
        self.e_i = self.e_i[..., newaxis]
        self.f_i = self.f_i[..., newaxis]
        self.IE_i = self.IE_i[..., newaxis]
        self.II_i = self.II_i[..., newaxis]
        self.m_i = self.m_i[..., newaxis]
        self.n_i = self.n_i[..., newaxis]


class ReducedSetHindmarshRose(ReducedSetBase):
    r"""
    .. [SJ_2008] Stefanescu and Jirsa, PLoS Computational Biology, *A Low
        Dimensional Description of Globally Coupled Heterogeneous Neural
        Networks of Excitatory and Inhibitory*  4, 11, 26--36, 2008.

    The models (:math:`\xi`, :math:`\eta`) phase-plane, including a
    representation of the vector field as well as its nullclines, using default
    parameters, can be seen below:

        .. _phase-plane-rHR_0:
        .. figure :: img/ReducedSetHindmarshRose_01_mode_0_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 1st mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the first mode of
            a reduced set of Hindmarsh-Rose oscillators.

        .. _phase-plane-rHR_1:
        .. figure :: img/ReducedSetHindmarshRose_01_mode_1_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 2nd mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the second mode of
            a reduced set of Hindmarsh-Rose oscillators.

        .. _phase-plane-rHR_2:
        .. figure :: img/ReducedSetHindmarshRose_01_mode_2_pplane.svg
            :alt: Reduced set of FitzHughNagumo phase plane (xi, eta), 3rd mode.

            The (:math:`\xi`, :math:`\eta`) phase-plane for the third mode of
            a reduced set of Hindmarsh-Rose oscillators.


    The dynamic equations were orginally taken from [SJ_2008]_.

    The equations of the population model for i-th mode at node q are:

    .. math::
                \dot{\xi}_i     &=  \eta_i-a_i\xi_i^3 + b_i\xi_i^2- \tau_i
                                 + K_{11} \left[\sum_{k=1}^{o} A_{ik} \xi_k - \xi_i \right]
                                 - K_{12} \left[\sum_{k=1}^{o} B_{ik} \alpha_k - \xi_i\right] + IE_i \\
                                &\, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                 + \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr} \right] \\
                & \\
                \dot{\eta}_i    &=  c_i-d_i\xi_i^2 -\tau_i \\
                & \\
                \dot{\tau}_i    &=  rs\xi_i - r\tau_i -m_i \\
                & \\
                \dot{\alpha}_i  &=  \beta_i - e_i \alpha_i^3 + f_i \alpha_i^2 - \gamma_i
                                 + K_{21} \left[\sum_{k=1}^{o} C_{ik} \xi_k - \alpha_i \right] + II_i \\
                                &\, +\left[\sum_{k=1}^{o}\mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                 + \left[\sum_{k=1}^{o}W_{\zeta}\cdot\xi_{kr}\right] \\
                & \\
                \dot{\beta}_i   &= h_i - p_i \alpha_i^2 - \beta_i \\
                \dot{\gamma}_i  &= rs \alpha_i - r \gamma_i - n_i

    .. automethod:: ReducedSetHindmarshRose.update_derived_parameters

    #NOTE: In the Article this modelis called StefanescuJirsa3D

    """

    # Define traited attributes for this model, these represent possible kwargs.
    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.006]),
        domain=Range(lo=0.0, hi=0.1, step=0.0005),
        doc="""Adaptation parameter""")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Dimensionless parameter as in the Hindmarsh-Rose model""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([3.0]),
        domain=Range(lo=0.0, hi=3.0, step=0.01),
        doc="""Dimensionless parameter as in the Hindmarsh-Rose model""")

    c = NArray(
        label=":math:`c`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Dimensionless parameter as in the Hindmarsh-Rose model""")

    d = NArray(
        label=":math:`d`",
        default=numpy.array([5.0]),
        domain=Range(lo=2.5, hi=7.5, step=0.01),
        doc="""Dimensionless parameter as in the Hindmarsh-Rose model""")

    s = NArray(
        label=":math:`s`",
        default=numpy.array([4.0]),
        domain=Range(lo=2.0, hi=6.0, step=0.01),
        doc="""Adaptation paramters, governs feedback""")

    xo = NArray(
        label=":math:`x_{o}`",
        default=numpy.array([-1.6]),
        domain=Range(lo=-2.4, hi=-0.8, step=0.01),
        doc="""Leftmost equilibrium point of x""")

    K11 = NArray(
        label=":math:`K_{11}`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, excitatory to excitatory""")

    K12 = NArray(
        label=":math:`K_{12}`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, inhibitory to excitatory""")

    K21 = NArray(
        label=":math:`K_{21}`",
        default=numpy.array([0.15]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Internal coupling, excitatory to inhibitory""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.3]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Standard deviation of Gaussian distribution""")

    mu = NArray(
        label=r":math:`\mu`",
        default=numpy.array([3.3]),
        domain=Range(lo=1.1, hi=3.3, step=0.01),
        doc="""Mean of Gaussian distribution""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"xi": numpy.array([-4.0, 4.0]),
                 "eta": numpy.array([-25.0, 20.0]),
                 "tau": numpy.array([2.0, 10.0]),
                 "alpha": numpy.array([-4.0, 4.0]),
                 "beta": numpy.array([-20.0, 20.0]),
                 "gamma": numpy.array([2.0, 10.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("xi", "eta", "tau", "alpha", "beta", "gamma"),
        default=("xi", "eta", "tau"),
        doc=r"""This represents the default state-variables of this Model to be
                monitored. It can be overridden for each Monitor if desired. The
                corresponding state-variable indices for this model are :math:`\xi = 0`,
                :math:`\eta = 1`, :math:`\tau = 2`, :math:`\alpha = 3`,
                :math:`\beta = 4`, and :math:`\gamma = 5`""")

    state_variables = 'xi eta tau alpha beta gamma'.split()
    _nvar = 6
    cvar = numpy.array([0, 3], dtype=numpy.int32)
    # derived parameters
    A_ik = None
    B_ik = None
    C_ik = None
    a_i = None
    b_i = None
    c_i = None
    d_i = None
    e_i = None
    f_i = None
    h_i = None
    p_i = None
    IE_i = None
    II_i = None
    m_i = None
    n_i = None

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The equations of the population model for i-th mode at node q are:

        .. math::
                \dot{\xi}_i     &=  \eta_i-a_i\xi_i^3 + b_i\xi_i^2- \tau_i
                                 + K_{11} \left[\sum_{k=1}^{o} A_{ik} \xi_k - \xi_i \right]
                                 - K_{12} \left[\sum_{k=1}^{o} B_{ik} \alpha_k - \xi_i\right] + IE_i \\
                                &\, + \left[\sum_{k=1}^{o} \mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                 + \left[\sum_{k=1}^{o} W_{\zeta}\cdot\xi_{kr} \right] \\
                & \\
                \dot{\eta}_i    &=  c_i-d_i\xi_i^2 -\tau_i \\
                & \\
                \dot{\tau}_i    &=  rs\xi_i - r\tau_i -m_i \\
                & \\
                \dot{\alpha}_i  &=  \beta_i - e_i \alpha_i^3 + f_i \alpha_i^2 - \gamma_i
                                 + K_{21} \left[\sum_{k=1}^{o} C_{ik} \xi_k - \alpha_i \right] + II_i \\
                                &\, +\left[\sum_{k=1}^{o}\mathbf{\Gamma}(\xi_{kq}, \xi_{kr}, u_{qr})\right]
                                 + \left[\sum_{k=1}^{o}W_{\zeta}\cdot\xi_{kr}\right] \\
                & \\
                \dot{\beta}_i   &= h_i - p_i \alpha_i^2 - \beta_i \\
                \dot{\gamma}_i  &= rs \alpha_i - r \gamma_i - n_i

        """

        xi = state_variables[0, :]
        eta = state_variables[1, :]
        tau = state_variables[2, :]
        alpha = state_variables[3, :]
        beta = state_variables[4, :]
        gamma = state_variables[5, :]
        derivative = numpy.empty_like(state_variables)

        c_0 = coupling[0, :].sum(axis=1)[:, numpy.newaxis]
        # c_1 = coupling[1, :]


        # for batching, we convert dot to einsum, so
        # dot(xi, Aik) results in an error of
        # ValueError: shapes (32,3,8) and (3,3,1) not aligned
        # instead use an einsum
        # numpy.einsum('nmb,mMo->nMb', xi, self.Aik)

        derivative[0] = (eta - self.a_i * xi ** 3 + self.b_i * xi ** 2 - tau +
               self.K11 * (numpy.einsum('nmb,mMo->nMb', xi, self.A_ik) - xi) -
               self.K12 * (numpy.einsum('nmb,mMo->nMb', alpha, self.B_ik) - xi) +
               self.IE_i + c_0 + local_coupling * xi)

        derivative[1] = self.c_i - self.d_i * xi ** 2 - eta

        derivative[2] = self.r * self.s * xi - self.r * tau - self.m_i

        derivative[3] = (beta - self.e_i * alpha ** 3 + self.f_i * alpha ** 2 - gamma +
                  self.K21 * (numpy.einsum('nmb,mMo->nMb', xi, self.C_ik) - alpha) +
                  self.II_i + c_0 + local_coupling * xi)

        derivative[4] = self.h_i - self.p_i * alpha ** 2 - beta

        derivative[5] = self.r * self.s * alpha - self.r * gamma - self.n_i

        return derivative

    def update_derived_parameters(self, corrected_d_p=True):
        """
        Calculate coefficients for the neural field model based on a Reduced set
        of Hindmarsh-Rose oscillators. Specifically, this method implements
        equations for calculating coefficients found in the supplemental
        material of [SJ_2008]_.

        Include equations here...

        """

        newaxis = numpy.newaxis
        from scipy.integrate import trapezoid as trapz
        from scipy.stats import norm as scipy_stats_norm

        stepu = 1.0 / (self.nu + 2 - 1)
        stepv = 1.0 / (self.nv + 2 - 1)

        norm = scipy_stats_norm(loc=self.mu, scale=self.sigma)

        Iu = norm.ppf(numpy.arange(stepu, 1.0, stepu))
        Iv = norm.ppf(numpy.arange(stepv, 1.0, stepv))

        # Define the modes
        V = numpy.zeros((self.number_of_modes, self.nv))
        U = numpy.zeros((self.number_of_modes, self.nu))

        nv_per_mode = self.nv // self.number_of_modes
        nu_per_mode = self.nu // self.number_of_modes

        for i in range(self.number_of_modes):
            V[i, i * nv_per_mode:(i + 1) * nv_per_mode] = numpy.ones(nv_per_mode)
            U[i, i * nu_per_mode:(i + 1) * nu_per_mode] = numpy.ones(nu_per_mode)

        # Normalise the modes
        V = V / numpy.tile(numpy.sqrt(trapz(V * V, Iv, axis=1)), (self.nv, 1)).T
        U = U / numpy.tile(numpy.sqrt(trapz(U * U, Iu, axis=1)), (self.nu, 1)).T

        # Get Normal PDF's evaluated with sampling Zv and Zu
        g1 = norm.pdf(Iv)
        g2 = norm.pdf(Iu)
        G1 = numpy.tile(g1, (self.number_of_modes, 1))
        G2 = numpy.tile(g2, (self.number_of_modes, 1))

        cV = numpy.conj(V)
        cU = numpy.conj(U)

        #import pdb; pdb.set_trace()
        intcVdI = trapz(cV, Iv, axis=1)[:, newaxis]
        intG1VdI = trapz(G1 * V, Iv, axis=1)[newaxis, :]
        intcUdI = trapz(cU, Iu, axis=1)[:, newaxis]

        #Calculate coefficients
        self.A_ik = numpy.dot(intcVdI, intG1VdI).T
        self.B_ik = numpy.dot(intcVdI, trapz(G2 * U, Iu, axis=1)[newaxis, :])
        self.C_ik = numpy.dot(intcUdI, intG1VdI).T

        self.a_i = self.a * trapz(cV * V ** 3, Iv, axis=1)[newaxis, :]
        self.e_i = self.a * trapz(cU * U ** 3, Iu, axis=1)[newaxis, :]
        self.b_i = self.b * trapz(cV * V ** 2, Iv, axis=1)[newaxis, :]
        self.f_i = self.b * trapz(cU * U ** 2, Iu, axis=1)[newaxis, :]
        self.c_i = (self.c * intcVdI).T
        self.h_i = (self.c * intcUdI).T

        self.IE_i = trapz(Iv * cV, Iv, axis=1)[newaxis, :]
        self.II_i = trapz(Iu * cU, Iu, axis=1)[newaxis, :]

        if corrected_d_p:
            # correction identified by Shrey Dutta & Arpan Bannerjee, confirmed by RS
            self.d_i = self.d * trapz(cV * V ** 2, Iv, axis=1)[newaxis, :]
            self.p_i = self.d * trapz(cU * U ** 2, Iu, axis=1)[newaxis, :]
        else:
            # typo in the original paper by RS & VJ, kept for comparison purposes.
            self.d_i = (self.d * intcVdI).T
            self.p_i = (self.d * intcUdI).T

        self.m_i = (self.r * self.s * self.xo * intcVdI).T
        self.n_i = (self.r * self.s * self.xo * intcUdI).T

        # batching requires to add a last axis of size 1
        self.A_ik = self.A_ik[..., newaxis]
        self.B_ik = self.B_ik[..., newaxis]
        self.C_ik = self.C_ik[..., newaxis]
        self.a_i = self.a_i[..., newaxis]
        self.b_i = self.b_i[..., newaxis]
        self.c_i = self.c_i[..., newaxis]
        self.d_i = self.d_i[..., newaxis]
        self.e_i = self.e_i[..., newaxis]
        self.f_i = self.f_i[..., newaxis]
        self.h_i = self.h_i[..., newaxis]
        self.p_i = self.p_i[..., newaxis]
        self.IE_i = self.IE_i[..., newaxis]
        self.II_i = self.II_i[..., newaxis]
        self.m_i = self.m_i[..., newaxis]
        self.n_i = self.n_i[..., newaxis]