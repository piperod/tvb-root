# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
The original Wong and Wang model
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <Marmaduke@tvb.invalid>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
"""


# Third party python libraries
import numpy

#The Virtual Brain
from tvb.simulator.common import get_logger
LOG = get_logger(__name__)

from tvb.basic.neotraits.api import NArray, Range, List, Final
import tvb.simulator.models as models

#import tvb.datatypes.arrays as arrays
#import tvb.basic.traits.types_basic as basic 
import tvb.simulator.models as models

class WongWang(models.Model):
    """
    .. [WW_2006] Kong-Fatt Wong and Xiao-Jing Wang,  *A Recurrent Network 
                Mechanism of Time Integration in Perceptual Decisions*. 
                Journal of Neuroscience 26(4), 1314-1328, 2006.
    .. [WW_2006_SI] Supplementary Information
    A reduced model by Wong and Wang: A reduced two-variable neural model 
    that offers a simple yet biophysically plausible framework for studying 
    perceptual decision making in general.
    S is the NMDA gating variable. Since its decay time is much longer that those
    corresponding to AMPAand GABA gating variables, it is assumed that is 
    :math:`S_{NMDA}` that dominates the time evolution of the system.
    The model (:math:`S1`, :math:`S2`) phase-plane, including a representation 
    of the vector field as well as its nullclines, using default parameters, 
    can be seen below:
    .. figure :: img/WongWang_01_mode_0_pplane.svg
    .. _phase-plane-WongWang:
        :alt: Phase plane of the reduced model by Wong and Wang (S1, S2)
    To reproduce the phase plane in Figure 4A, page 1319 (five steady states):
        J11 = 0.54
        J22 = 0.18
        J12 = 0.08
        J21 = 0.58
        J_ext = 0.0
        I_o = 0.34
        sigma_noise = 0.02
        mu_o = 0.00
        c = 100.0
    To reproduce the phase plane in Figure 4B, page 1319 (saddle-type point):
        b = 0.108
        d = 121.0
        gamma = 0.11
        tau_s = 100.
        J11 = 0.78
        J22 = 0.59
        J12 = 0.72
        J21 = 0.67
        J_ext = 0.52
        I_o = 0.3255
        sigma_noise = 0.02
        mu_o = 0.35
        c = 0.0
    .. automethod:: __init__
    """
    _ui_name = "Wong-Wang (Original)"
            
               
    #Define traited attributes for this model, these represent possible kwargs.
    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.270, ]),
        domain=Range(lo=0.260, hi=0.272),
        doc=""" (mVnC)^{-1}. Parameter chosen to ﬁt numerical solutions.""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.108, ]),
        domain=Range(lo=0.1070001, hi=0.109001),
        doc="""[kHz]. Parameter chosen to ﬁt numerical solutions.""")

    d = NArray(
        label=":math:`d`",
        default=numpy.array([154, ]),#154
        domain=Range(lo=0.0, hi=155),
        doc="""[ms]. Parameter chosen to ﬁt numerical solutions.""")

    gamma = NArray(
        label=r":math:`\gamma`",
        default=numpy.array([0.641, ]),#0.0641
        domain=Range(lo=0.0, hi=1.0),
        doc="""Kinetic parameter divided by 1000 to set the time scale in ms""")

    tau_s = NArray(
        label=r":math:`\tau_S`",
        default=numpy.array([100., ]),
        domain=Range(lo=50.0, hi=150.0),
        doc="""Kinetic parameter. NMDA decay time constant.""")

    tau_ampa = NArray(
        label=r":math:`\tau_{ampa}`",
        default=numpy.array([2., ]),
        domain=Range(lo=1.0, hi=10.0),
        doc="""Kinetic parameter. AMPA decay time constant.""",
       )

    J11 = NArray(
        label=":math:`J_{11}`",
        default=numpy.array([0.2609, ]),#0.2609
        domain=Range(lo=0.0, hi=1.0),
        doc="""Synaptic coupling""")

    J22 = NArray(
        label=":math:`J_{22}`",
        default=numpy.array([0.2609, ]),#0.2609
        domain=Range(lo=0.0, hi=1.0),
        doc="""Synaptic coupling""")

    J12 = NArray(
        label=":math:`J_{12}`",
        default=numpy.array([0.1, ]),#0.0497
        domain=Range(lo=0.0, hi=1.0),
        doc="""Synaptic coupling""")

    J21 = NArray(
        label=":math:`J_{21}`",
        default=numpy.array([0.1, ]),#0.0497
        domain=Range(lo=0.0, hi=1.0),
        doc="""Synaptic coupling""")

    J_ext = NArray(
        label=":math:`J_{ext}`",
        default=numpy.array([0.00052, ]),
        domain=Range(lo=0.0, hi=1.0),
        doc="""Synaptic coupling""")

    I_o = NArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.28, ]),
        domain=Range(lo=0.0, hi=1.0),
        doc="""Effective external input""")

    sigma_noise = NArray(
        label=r":math:`\sigma_{noise}`",
        default=numpy.array([0.02, ]),
        domain=Range(lo=0.0, hi=1.0),
        doc="""Noise amplitude. Take this value into account for stochatic
        integration schemes.""")

    mu_o = NArray(
        label=r":math:`\mu_{0}`",
        default=numpy.array([30, ]),#0.03
        domain=Range(lo=10, hi=50),
        doc="""Stimulus amplitude""")

    c = NArray(
        label=":math:`c`",
        default=numpy.array([0.21, ]),#52
        domain=Range(lo=0.0, hi=1.0),
        doc="""[%].  Percentage coherence or motion strength. This parameter
        comes from experiments in MT cells.""")

    state_variable_range = Final(
        {
             "S1": numpy.array([0, 1]),
             "S2": numpy.array([0, 1])},
        label="State variable ranges [lo, hi]",
        doc="n/a"
    )
    state_variables = ['S1', 'S2']


    variables_of_interest = List(
        of = str,
        label="Variables watched by Monitors",
        choices=["S1", "S2"],
        default=["S1"],
        #select_multiple=True,
        doc="""default state variables to be monitored""",
        )#order=10)


    def __init__(self, **kwargs):
        """
        .. May need to put kwargs back if we can't get them from trait...
        """

        #LOG.info('%s: initing...' % str(self))

        super(WongWang, self).__init__(**kwargs)

        #self._state_variables = ["S1", "S2"]
        self._nvar = 2
        self.cvar = numpy.array([0], dtype=numpy.int32)

        #derived parameters
        self.I_1 = None
        self.I_2 = None

        LOG.debug('%s: inited.' % repr(self))

    def configure(self):
        """  """
        super(WongWang, self).configure()
        self.update_derived_parameters()

    # def ornstein_uhlenbeck(self,T,tau,sigma=0.009,x0=0,dt=1):
    #     yita = numpy.random.normal(T/dt,0,numpy.sqrt(dt))
    #     n = T/dt
    #     x = numpy.zeros(n)
    #     x[0]=x0
    #     for i in range(2,n+1):
    #         x[i] = x[i-1] + (-x[i-1]*dt/tau + sigma*yita[i-1]*dt/numpy.sqrt(tau))
    #     return x

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        These dynamic equations, taken from [WW_2006]_, ...
        ..math::
            \frac{dS_{i}}{dt} &= - \frac{S_{i}}{\tau_{S}} + (1 - S_{i})\gamma H_{i} \\
            H_{i} &= \frac{a x_{i} - b}{1- \exp[-d (a x_{i} - b)]} \\
            x_{1} &= J11  S_{1} - J_{12}S_{2} + I_{0} + I_{1} \\
            x_{2} &= J22  S_{2} - J_{21}S_{1} + I_{0} + I_{2} \\
            I_{i} &= J_{ext} \mu_{0} \left( 1 \pm \frac{c}{100}\right)
        where :math:`i=` 1, 2 labels the selective population.
        """
        # add global coupling?
        s1 = state_variables[0, :]
        s2 = state_variables[1, :]

        c_0 = coupling[0]
        #import pdb;pdb.set_trace()
        x1 = self.J11 * s1 - self.J12 * s2 + self.I_o + self.I_1 
        #x2 = self.J21 * s2 - self.J22 * s1 + self.I_o + self.I_2
        x2 = self.J22 * s2 - self.J21 * s1 + self.I_o + self.I_2

        H1 = (self.a * x1 - self.b) / (1 - numpy.exp(-self.d * (self.a * x1 - \
                                                                self.b)))
        H2 = (self.a * x2 - self.b) / (1 - numpy.exp(-self.d * (self.a * x2 - \
                                                                self.b)))

        ds1 = - (s1 / self.tau_s) + (1 - s1) * H1 * self.gamma
        ds2 = - (s2 / self.tau_s) + (1 - s2) * H2 * self.gamma

        derivative = numpy.array([ds1, ds2])

        return derivative


    def update_derived_parameters(self):
        """
        Derived parameters
        """

        self.I_1 = self.J_ext * self.mu_o * (1 + self.c )
        self.I_2 = self.J_ext * self.mu_o * (1 - self.c )



if __name__ == "__main__":
    # Do some stuff that tests or makes use of this module...
    LOG.info("Testing %s module..." % __file__)
    
    # Check that the docstring examples, if there are any, are accurate.
    import doctest
    doctest.testmod()
    
    #Initialise Models in their default state:
    WW = WongWang()
        
    LOG.info("Model initialised in its default state without error...")
    
    LOG.info("Testing phase plane interactive ... ")
    


    
    ## Check the Phase Plane
    from tvb.simulator.plot.phase_plane_interactive import PhasePlaneInteractive
    import tvb.simulator.integrators
    import tvb.simulator.noise
    import pyargs
    import numpy 
    import matplotlib.pyplot as plt
    noise = tvb.simulator.noise.Additive()
    INTEGRATOR = tvb.simulator.integrators.HeunDeterministic(dt=1)
    scheme = INTEGRATOR.scheme
    svx = WW.state_variables[0]
    svy = WW.state_variables[1]
    svx_ind = WW.state_variables.index(svx)
    svy_ind = WW.state_variables.index(svy)
    sv_mean = numpy.array([WW.state_variable_range[key].mean() for key in WW.state_variables])
    sv_mean = sv_mean.reshape((WW.nvar, 1, 1))
    default_sv = sv_mean.repeat(WW.number_of_modes,axis=2)
    no_coupling = numpy.zeros((WW.nvar, 1,
                                        WW.number_of_modes))
    INTEGRATOR.clamped_state_variable_indices = numpy.setdiff1d(
            numpy.r_[:len(WW.state_variables)], numpy.r_[svx_ind, svy_ind])
    state = default_sv.copy()
    # dataset =[]
    # threshold = 1.5e-18
    # counter = 0
    # for x in numpy.arange(0,1,0.01):
    #     for y in numpy.arange(0,1,0.01):
    #         counter+=1
    #         state[svx_ind] = x
    #         state[svy_ind] = y
    #         TRAJ_STEPS=4096
    #         traj = numpy.zeros((TRAJ_STEPS+1, WW.nvar, 1,
    #                         WW.number_of_modes))
    #         traj[0, :] = state
    #         WW.update_derived_parameters()
    #         for step in range(TRAJ_STEPS):
    #             #import pdb; pdb.set_trace()
    #             state = scheme(state, WW.dfun, no_coupling, 0.0, 0.0)
    #             traj[step+1, :] = state
    #         xx = traj[:, svx_ind, 0, 0]
    #         yy = traj[:, svy_ind, 0, 0]
    #         distances  = numpy.square((xx[:-1]-xx[1:])**2 + (yy[:-1]-yy[1:])**2)
    #         steps = (distances >threshold).sum()
    #         if counter%10==0: 
    #             print(counter)
    #             print(distances)
    #             print((distances > threshold).sum())
    #         dataset.append([x,y,xx[-1],yy[-1],steps,threshold,xx,yy,distances])

    # import pandas as pd 

    # datadf = pd.DataFrame(dataset,columns=['S1_i','S2_i','S1_f','S_2_f','steps','threshold_distances','raw s1','raw s2','raw_distances'])
    # datadf.to_csv('dataset_activations.csv')
    # plt.scatter(x,y,s=42,c='g',marker='o')
    
    # plt.scatter(xx,yy)
    # plt.show()
    # plt.scatter(range(len(distances)),distances)
    # plt.show()
    #import pdb;pdb.set_trace()
    #print('initializing')
    ppi_fig = PhasePlaneInteractive(model=WW, integrator=INTEGRATOR)
    ppi_fig.show()