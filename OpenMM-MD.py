#!/usr/bin/env python

"""
@package run

Run a MD simulation in OpenMM.  NumPy is required.

Copyright And License

@author Lee-Ping Wang <leeping@stanford.edu>

All code in this file is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but without any
warranty; without even the implied warranty of merchantability or fitness for a
particular purpose.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

"""

#==================#
#| Global Imports |#
#==================#

import time
from datetime import datetime, timedelta
t0 = time.time()
from ast import literal_eval as leval
import argparse
from xml.etree import ElementTree as ET
import os
import sys
import pickle
import shutil
import numpy as np
from re import sub
from collections import namedtuple, defaultdict, OrderedDict
from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *
import warnings
# Suppress warnings from PDB reading.
warnings.simplefilter("ignore")
import logging
logging.basicConfig()

#================================#
#       Set up the logger        #
#================================#

logger = logging.getLogger(__name__)
logger.setLevel('INFO')
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt="%H:%M:%S")
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.propagate = False

#================================#
#          Subroutines           #
#================================#

def GetTime(sec):
    sec = timedelta(seconds=sec)
    d = datetime(1,1,1) + sec
    if d.year > 1:
        return("%dY%02dM%02dd%02dh%02dm%02ds" % (d.year-1, d.month-1, d.day-1, d.hour, d.minute, d.second))
    elif d.month > 1:
        return("%dM%02dd%02dh%02dm%02ds" % (d.month-1, d.day-1, d.hour, d.minute, d.second))
    elif d.day > 1:
        return("%dd%02dh%02dm%02ds" % (d.day-1, d.hour, d.minute, d.second))
    elif d.hour > 0: 
        return("%dh%02dm%02ds" % (d.hour, d.minute, d.second))
    elif d.minute > 0: 
        return("%dm%02ds" % (d.minute, d.second))
    elif d.second > 0: 
        return("%ds" % (d.second))

def statisticalInefficiency(A_n, B_n=None, fast=False, mintime=3):

    """
    Compute the (cross) statistical inefficiency of (two) timeseries.

    Notes
      The same timeseries can be used for both A_n and B_n to get the autocorrelation statistical inefficiency.
      The fast method described in Ref [1] is used to compute g.

    References
      [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
      histogram analysis method for the analysis of simulated and parallel tempering simulations.
      JCTC 3(1):26-41, 2007.

    Examples

    Compute statistical inefficiency of timeseries data with known correlation time.

    >>> import timeseries
    >>> A_n = timeseries.generateCorrelatedTimeseries(N=100000, tau=5.0)
    >>> g = statisticalInefficiency(A_n, fast=True)

    @param[in] A_n (required, numpy array) - A_n[n] is nth value of
    timeseries A.  Length is deduced from vector.

    @param[in] B_n (optional, numpy array) - B_n[n] is nth value of
    timeseries B.  Length is deduced from vector.  If supplied, the
    cross-correlation of timeseries A and B will be estimated instead of
    the autocorrelation of timeseries A.

    @param[in] fast (optional, boolean) - if True, will use faster (but
    less accurate) method to estimate correlation time, described in
    Ref. [1] (default: False)

    @param[in] mintime (optional, int) - minimum amount of correlation
    function to compute (default: 3) The algorithm terminates after
    computing the correlation time out to mintime when the correlation
    function furst goes negative.  Note that this time may need to be
    increased if there is a strong initial negative peak in the
    correlation function.

    @return g The estimated statistical inefficiency (equal to 1 + 2
    tau, where tau is the correlation time).  We enforce g >= 1.0.

    """

    # Create numpy copies of input arguments.
    A_n = np.array(A_n)
    if B_n is not None:
        B_n = np.array(B_n)
    else:
        B_n = np.array(A_n)
    # Get the length of the timeseries.
    N = A_n.size
    # Be sure A_n and B_n have the same dimensions.
    if(A_n.shape != B_n.shape):
        raise ParameterError('A_n and B_n must have same dimensions.')
    # Initialize statistical inefficiency estimate with uncorrelated value.
    g = 1.0
    # Compute mean of each timeseries.
    mu_A = A_n.mean()
    mu_B = B_n.mean()
    # Make temporary copies of fluctuation from mean.
    dA_n = A_n.astype(np.float64) - mu_A
    dB_n = B_n.astype(np.float64) - mu_B
    # Compute estimator of covariance of (A,B) using estimator that will ensure C(0) = 1.
    sigma2_AB = (dA_n * dB_n).mean() # standard estimator to ensure C(0) = 1
    # Trap the case where this covariance is zero, and we cannot proceed.
    if(sigma2_AB == 0):
        logger.info('Sample covariance sigma_AB^2 = 0 -- cannot compute statistical inefficiency')
        return 1.0
    # Accumulate the integrated correlation time by computing the normalized correlation time at
    # increasing values of t.  Stop accumulating if the correlation function goes negative, since
    # this is unlikely to occur unless the correlation function has decayed to the point where it
    # is dominated by noise and indistinguishable from zero.
    t = 1
    increment = 1
    while (t < N-1):
        # compute normalized fluctuation correlation function at time t
        C = sum( dA_n[0:(N-t)]*dB_n[t:N] + dB_n[0:(N-t)]*dA_n[t:N] ) / (2.0 * float(N-t) * sigma2_AB)
        # Terminate if the correlation function has crossed zero and we've computed the correlation
        # function at least out to 'mintime'.
        if (C <= 0.0) and (t > mintime):
            break
        # Accumulate contribution to the statistical inefficiency.
        g += 2.0 * C * (1.0 - float(t)/float(N)) * float(increment)
        # Increment t and the amount by which we increment t.
        t += increment
        # Increase the interval if "fast mode" is on.
        if fast: increment += 1
    # g must be at least unity
    if (g < 1.0): g = 1.0
    # Return the computed statistical inefficiency.
    return g

def compute_volume(box_vectors):
    """ Compute the total volume of an OpenMM system. """
    [a,b,c] = box_vectors
    A = np.array([a/a.unit, b/a.unit, c/a.unit])
    # Compute volume of parallelepiped.
    volume = np.linalg.det(A) * a.unit**3
    return volume

def compute_mass(system):
    """ Compute the total mass of an OpenMM system. """
    mass = 0.0 * amu
    for i in range(system.getNumParticles()):
        mass += system.getParticleMass(i)
    return mass

def printcool(text,sym="#",bold=False,color=2,ansi=None,bottom='-',minwidth=50):
    """Cool-looking printout for slick formatting of output.

    @param[in] text The string that the printout is based upon.  This function
    will print out the string, ANSI-colored and enclosed in the symbol
    for example:\n
    <tt> ################# </tt>\n
    <tt> ### I am cool ### </tt>\n
    <tt> ################# </tt>
    @param[in] sym The surrounding symbol\n
    @param[in] bold Whether to use bold print
    
    @param[in] color The ANSI color:\n
    1 red\n
    2 green\n
    3 yellow\n
    4 blue\n
    5 magenta\n
    6 cyan\n
    7 white
    
    @param[in] bottom The symbol for the bottom bar

    @param[in] minwidth The minimum width for the box, if the text is very short
    then we insert the appropriate number of padding spaces

    @return bar The bottom bar is returned for the user to print later, e.g. to mark off a 'section'    
    """
    if logger.getEffectiveLevel() < 20: return
    def newlen(l):
        return len(sub("\x1b\[[0-9;]*m","",line))
    text = text.split('\n')
    width = max(minwidth,max([newlen(line) for line in text]))
    bar = ''.join([sym for i in range(width + 8)])
    print '\n'+bar
    for line in text:
        padleft = ' ' * ((width - newlen(line)) / 2)
        padright = ' '* (width - newlen(line) - len(padleft))
        if ansi != None:
            ansi = str(ansi)
            print "%s| \x1b[%sm%s" % (sym, ansi, padleft),line,"%s\x1b[0m |%s" % (padright, sym)
        elif color != None:
            print "%s| \x1b[%s9%im%s" % (sym, bold and "1;" or "", color, padleft),line,"%s\x1b[0m |%s" % (padright, sym)
        else:
            warn_press_key("Inappropriate use of printcool")
    print bar
    return sub(sym,bottom,bar)

def printcool_dictionary(Dict,title="General options",bold=False,color=2,keywidth=25,topwidth=50):
    """See documentation for printcool; this is a nice way to print out keys/values in a dictionary.

    The keys in the dictionary are sorted before printing out.

    @param[in] dict The dictionary to be printed
    @param[in] title The title of the printout
    """
    if logger.getEffectiveLevel() < 20: return
    if Dict == None: return
    bar = printcool(title,bold=bold,color=color,minwidth=topwidth)
    def magic_string(str):
        # This cryptic command returns a string with the number of characters specified as a variable. :P
        # Useful for printing nice-looking dictionaries, i guess.
        #print "\'%%-%is\' %% '%s'" % (keywidth,str.replace("'","\\'").replace('"','\\"'))
        return eval("\'%%-%is\' %% '%s'" % (keywidth,str.replace("'","\\'").replace('"','\\"')))
    if isinstance(Dict, OrderedDict): 
        print '\n'.join(["%s %s " % (magic_string(str(key)),str(Dict[key])) for key in Dict if Dict[key] != None])
    else:
        print '\n'.join(["%s %s " % (magic_string(str(key)),str(Dict[key])) for key in sorted([i for i in Dict]) if Dict[key] != None])
    print bar

def EnergyDecomposition(Sim, verbose=False):
    # Before using EnergyDecomposition, make sure each Force is set to a different group.
    EnergyTerms = OrderedDict()
    Potential = Sim.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
    Kinetic = Sim.context.getState(getEnergy=True).getKineticEnergy() / kilojoules_per_mole
    for i in range(Sim.system.getNumForces()):
        EnergyTerms[Sim.system.getForce(i).__class__.__name__] = Sim.context.getState(getEnergy=True,groups=2**i).getPotentialEnergy() / kilojoules_per_mole
    EnergyTerms['Potential'] = Potential
    EnergyTerms['Kinetic'] = Kinetic
    EnergyTerms['Total'] = Potential+Kinetic
    return EnergyTerms

def MTSVVVRIntegrator(temperature, collision_rate, timestep, system, ninnersteps=4):
    """
    Create a multiple timestep velocity verlet with velocity randomization (VVVR) integrator.
    
    ARGUMENTS

    temperature (numpy.unit.Quantity compatible with kelvin) - the temperature
    collision_rate (numpy.unit.Quantity compatible with 1/picoseconds) - the collision rate
    timestep (numpy.unit.Quantity compatible with femtoseconds) - the integration timestep
    system (simtk.openmm.System) - system whose forces will be partitioned
    ninnersteps (int) - number of inner timesteps (default: 4)

    RETURNS

    integrator (openmm.CustomIntegrator) - a VVVR integrator

    NOTES
    
    This integrator is equivalent to a Langevin integrator in the velocity Verlet discretization with a
    timestep correction to ensure that the field-free diffusion constant is timestep invariant.  The inner
    velocity Verlet discretization is transformed into a multiple timestep algorithm.

    REFERENCES

    VVVR Langevin integrator: 
    * http://arxiv.org/abs/1301.3800
    * http://arxiv.org/abs/1107.2967 (to appear in PRX 2013)    
    
    TODO

    Move initialization of 'sigma' to setting the per-particle variables.
    
    """
    # Multiple timestep Langevin integrator.
    for i in system.getForces():
        if i.__class__.__name__ in ["NonbondedForce", "CustomNonbondedForce", "AmoebaVdwForce", "AmoebaMultipoleForce"]:
            # Slow force.
            print i.__class__.__name__, "is a Slow Force"
            i.setForceGroup(1)
        else:
            print i.__class__.__name__, "is a Fast Force"
            # Fast force.
            i.setForceGroup(0)

    kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
    kT = kB * temperature
    
    integrator = openmm.CustomIntegrator(timestep)
    
    integrator.addGlobalVariable("dt_fast", timestep/float(ninnersteps)) # fast inner timestep
    integrator.addGlobalVariable("kT", kT) # thermal energy
    integrator.addGlobalVariable("a", numpy.exp(-collision_rate*timestep)) # velocity mixing parameter
    integrator.addGlobalVariable("b", numpy.sqrt((2/(collision_rate*timestep)) * numpy.tanh(collision_rate*timestep/2))) # timestep correction parameter
    integrator.addPerDofVariable("sigma", 0) 
    integrator.addPerDofVariable("x1", 0) # position before application of constraints

    #
    # Pre-computation.
    # This only needs to be done once, but it needs to be done for each degree of freedom.
    # Could move this to initialization?
    #
    integrator.addComputePerDof("sigma", "sqrt(kT/m)")

    # 
    # Velocity perturbation.
    #
    integrator.addComputePerDof("v", "sqrt(a)*v + sqrt(1-a)*sigma*gaussian")
    integrator.addConstrainVelocities();
    
    #
    # Symplectic inner multiple timestep.
    #
    integrator.addUpdateContextState(); 
    integrator.addComputePerDof("v", "v + 0.5*b*dt*f1/m")
    for innerstep in range(ninnersteps):
        # Fast inner symplectic timestep.
        integrator.addComputePerDof("v", "v + 0.5*b*dt_fast*f0/m")
        integrator.addComputePerDof("x", "x + v*b*dt_fast")
        integrator.addComputePerDof("x1", "x")
        integrator.addConstrainPositions();        
        integrator.addComputePerDof("v", "v + 0.5*b*dt_fast*f0/m + (x-x1)/dt_fast")
    integrator.addComputePerDof("v", "v + 0.5*b*dt*f1/m") # TODO: Additional velocity constraint correction?
    integrator.addConstrainVelocities();

    #
    # Velocity randomization
    #
    integrator.addComputePerDof("v", "sqrt(a)*v + sqrt(1-a)*sigma*gaussian")
    integrator.addConstrainVelocities();

    return integrator

def bak(fnm):
    oldfnm = fnm
    if os.path.exists(oldfnm):
        base, ext = os.path.splitext(fnm)
        i = 1
        while os.path.exists(fnm):
            fnm = "%s_%i%s" % (base,i,ext)
            i += 1
        logger.info("Backing up %s -> %s" % (oldfnm, fnm))
        shutil.move(oldfnm,fnm)

#================================#
#     The input file parser      #
#================================#

class SimulationOptions(object):
    """ Class for parsing the input file. """
    def set_active(self,key,default,typ,doc,allowed=None,depend=True,clash=False,msg=None):
        """ Set one option.  The arguments are:
        key     : The name of the option.
        default : The default value.
        typ     : The type of the value.
        doc     : The documentation string.
        allowed : An optional list of allowed values.
        depend  : A condition that must be True for the option to be activated.
        clash   : A condition that must be False for the option to be activated.
        msg     : A warning that is printed out if the option is not activated.
        """
        doc = sub("\.$","",doc.strip())+"."
        self.Documentation[key] = "%-8s " % ("(" + sub("'>","",sub("<type '","",str(typ)))+")") + doc
        if key in self.UserOptions:
            val = self.UserOptions[key]
        else:
            val = default
        if type(allowed) is list:
            self.Documentation[key] += " Allowed values are %s" % str(allowed)
            if val not in allowed:
                raise Exception("Tried to set option \x1b[1;91m%s\x1b[0m to \x1b[94m%s\x1b[0m but it's not allowed (choose from \x1b[92m%s\x1b[0m)" % (key, str(val), str(allowed)))
        if typ is bool and type(val) == int:
            val = bool(val)
        if val != None and type(val) is not typ:
            raise Exception("Tried to set option \x1b[1;91m%s\x1b[0m to \x1b[94m%s\x1b[0m but it's not the right type (%s required)" % (key, str(val), str(typ)))
        if depend and not clash:
            if key in self.InactiveOptions:
                del self.InactiveOptions[key]
            self.ActiveOptions[key] = val
        else:
            if key in self.ActiveOptions:
                del self.ActiveOptions[key]
            self.InactiveOptions[key] = val
            self.InactiveWarnings[key] = msg

    def force_active(self,key,val=None,msg=None):
        """ Force an option to be active and set it to the provided value,
        regardless of the user input.  There are no safeguards, so use carefully.
        
        key     : The name of the option.
        val     : The value that the option is being set to.
        msg     : A warning that is printed out if the option is not activated.
        """
        if msg == None:
            msg == "Option forced to active for no given reason."
        if key not in self.ActiveOptions:
            if val == None:
                val = self.InactiveOptions[key]
            del self.InactiveOptions[key]
            self.ActiveOptions[key] = val
            self.ForcedOptions[key] = val
            self.ForcedWarnings[key] = msg
        elif val != None and self.ActiveOptions[key] != val:
            self.ActiveOptions[key] = val
            self.ForcedOptions[key] = val
            self.ForcedWarnings[key] = msg
        elif val == None:
            self.ForcedOptions[key] = self.ActiveOptions[key]
            self.ForcedWarnings[key] = msg + " (Warning: Forced active but it was already active.)"
                
    def deactivate(self,key,msg=None):
        """ Deactivate one option.  The arguments are:
        key     : The name of the option.
        msg     : A warning that is printed out if the option is not activated.
        """
        if key in self.ActiveOptions:
            self.InactiveOptions[key] = self.ActiveOptions[key]
            del self.ActiveOptions[key]
        self.InactiveWarnings[key] = msg
        
    def __getattr__(self,key):
        if key in self.ActiveOptions:
            return self.ActiveOptions[key]
        elif key in self.InactiveOptions:
            return None
        else:
            return getattr(super(SimulationOptions,self),key)

    def record(self):
        out = []
        cmd = ' '.join(sys.argv)
        out.append("")
        out.append("Your command was: %s" % cmd)
        out.append("To reproduce / customize your simulation, paste the following text into an input file")
        out.append("and rerun the script with the -I argument (e.g. '-I openmm.in')")
        out.append("")
        out.append("#===========================================#")
        out.append("#|     Input file for OpenMM MD script     |#")
        out.append("#|  Lines beginning with '#' are comments  |#")
        out.append("#===========================================#")
        TopBar = False
        UserSupplied = []
        for key in self.ActiveOptions:
            if key in self.UserOptions and key not in self.ForcedOptions:
                UserSupplied.append("%-22s %20s # %s" % (key, str(self.ActiveOptions[key]), self.Documentation[key]))
        if len(UserSupplied) > 0:
            if TopBar:
                out.append("#===========================================#")
            else:
                TopBar = True
            out.append("#|          User-supplied options:         |#")
            out.append("#===========================================#")
            out += UserSupplied
        Forced = []
        for key in self.ActiveOptions:
            if key in self.ForcedOptions:
                Forced.append("%-22s %20s # %s" % (key, str(self.ActiveOptions[key]), self.Documentation[key]))
                Forced.append("%-22s %20s # Reason : %s" % ("","",self.ForcedWarnings[key]))
        if len(Forced) > 0:
            if TopBar:
                out.append("#===========================================#")
            else:
                TopBar = True
            out.append("#|     Options enforced by the script:     |#")
            out.append("#===========================================#")
            out += Forced
        ActiveDefault = []
        for key in self.ActiveOptions:
            if key not in self.UserOptions and key not in self.ForcedOptions:
                ActiveDefault.append("%-22s %20s # %s" % (key, str(self.ActiveOptions[key]), self.Documentation[key]))
        if len(ActiveDefault) > 0:
            if TopBar:
                out.append("#===========================================#")
            else:
                TopBar = True
            out.append("#|   Active options at default values:     |#")
            out.append("#===========================================#")
            out += ActiveDefault
        # out.append("")
        out.append("#===========================================#")
        out.append("#|           End of Input File             |#")
        out.append("#===========================================#")
        Deactivated = []
        for key in self.InactiveOptions:
            Deactivated.append("%-22s %20s # %s" % (key, str(self.InactiveOptions[key]), self.Documentation[key]))
            Deactivated.append("%-22s %20s # Reason : %s" % ("","",self.InactiveWarnings[key]))
        if len(Deactivated) > 0:
            out.append("")
            out.append("#===========================================#")
            out.append("#|   Deactivated or conflicting options:   |#")
            out.append("#===========================================#")
            out += Deactivated
        Unrecognized = []
        for key in self.UserOptions:
            if key not in self.ActiveOptions and key not in self.InactiveOptions:
                Unrecognized.append("%-22s %20s" % (key, self.UserOptions[key]))
        if len(Unrecognized) > 0:
            # out.append("")
            out.append("#===========================================#")
            out.append("#|          Unrecognized options:          |#")
            out.append("#===========================================#")
            out += Unrecognized
        return out
        
    def __init__(self, input_file, pdbfnm):
        super(SimulationOptions,self).__init__()
        basename = os.path.splitext(pdbfnm)[0]
        self.Documentation = OrderedDict()
        self.UserOptions = OrderedDict()
        self.ActiveOptions = OrderedDict()
        self.ForcedOptions = OrderedDict()
        self.ForcedWarnings = OrderedDict()
        self.InactiveOptions = OrderedDict()
        self.InactiveWarnings = OrderedDict()
        # First build a dictionary of user supplied options.
        if input_file != None:
            for line in open(input_file).readlines():
                line = sub('#.*$','',line.strip())
                s = line.split()
                if len(s) > 0:
                    # Options are case insensitive
                    key = s[0].lower()
                    try:
                        val = leval(line.replace(s[0],'',1).strip())
                    except:
                        val = str(line.replace(s[0],'',1).strip())
                    self.UserOptions[key] = val
        # Now go through the logic of determining which options are activated.
        self.set_active('integrator','verlet',str,"Molecular dynamics integrator",allowed=["verlet","langevin","velocity-verlet","mtsvvvr"])
        self.set_active('minimize',False,bool,"Specify whether to minimize the energy before running dynamics.")
        self.set_active('timestep',1.0,float,"Time step in femtoseconds.")
        self.set_active('innerstep',0.5,float,"Inner time step in femtoseconds for MTS integrator.",depend=(self.integrator=="mtsvvvr"))
        self.set_active('restart_filename','restart.p',str,"Restart information will be read from / written to this file (will be backed up).")
        self.set_active('read_restart',True,bool,"Restart simulation from the restart file.",
                        depend=(os.path.exists(self.restart_filename)), msg="Cannot restart; file specified by restart_filename does not exist.")
        self.set_active('restart_interval',1000,int,"Specify a timestep interval for writing the restart file.")
        self.set_active('equilibrate',0,int,"Number of steps reserved for equilibration.")
        self.set_active('production',1000,int,"Number of steps in production run.")
        self.set_active('report_interval',100,int,"Number of steps between every progress report.")
        self.set_active('temperature',0.0,float,"Simulation temperature for Langevin integrator or Andersen thermostat.")
        if self.temperature <= 0.0 and self.integrator in ["langevin", "mtsvvvr"]:
            raise Exception("You need to set a finite temperature if using the Langevin or MTS-VVVR integrator!")
        self.set_active('gentemp',self.temperature,float,"Specify temperature for generating velocities")
        self.set_active('collision_rate',0.1,float,"Collision frequency for Langevin integrator or Andersen thermostat in ps^-1.",
                        depend=(self.integrator in ["langevin", "mtsvvvr"] or self.temperature != 0.0),
                        msg="We're not running a constant temperature simulation")
        self.set_active('pressure',0.0,float,"Simulation pressure; set a positive number to activate.",
                        clash=(self.temperature <= 0.0),
                        msg="For constant pressure simulations, the temperature must be finite")
        self.set_active('anisotropic',False,bool,"Set to True for anisotropic box scaling in NPT simulations",
                        depend=("pressure" in self.ActiveOptions and self.pressure > 0.0), msg = "We're not running a constant pressure simulation")
        self.set_active('nbarostat',25,int,"Step interval for MC barostat volume adjustments.",
                        depend=("pressure" in self.ActiveOptions and self.pressure > 0.0), msg = "We're not running a constant pressure simulation")
        self.set_active('nonbonded_method','PME',str,"Set the method for nonbonded interactions.", allowed=["NoCutoff","CutoffNonPeriodic","CutoffPeriodic","Ewald","PME"])
        self.nonbonded_method_obj = {"NoCutoff":NoCutoff,"CutoffNonPeriodic":CutoffNonPeriodic,"CutoffPeriodic":CutoffPeriodic,"Ewald":Ewald,"PME":PME}[self.nonbonded_method]
        self.set_active('nonbonded_cutoff',0.9,float,"Nonbonded cutoff distance in nanometers.")
        self.set_active('vdw_switch',True,bool,"Use a multiplicative switching function to ensure twice-differentiable vdW energies near the cutoff distance.")
        self.set_active('switch_distance',0.8,float,"Set the distance where the switching function starts; must be less than the nonbonded cutoff.")
        self.set_active('dispersion_correction',True,bool,"Isotropic long-range dispersion correction for periodic systems.")
        self.set_active('ewald_error_tolerance',0.0,float,"Error tolerance for Ewald and PME methods.  Don't go below 5e-5 for PME unless running in double precision.",
                        depend=(self.nonbonded_method_obj in [Ewald, PME]), msg="Nonbonded method must be set to Ewald or PME.")
        self.set_active('platform',"CUDA",str,"The simulation platform.", allowed=["Reference","CUDA","OpenCL"])
        self.set_active('cuda_precision','single',str,"The precision of the CUDA platform.", allowed=["single","mixed","double"], 
                        depend=(self.platform == "CUDA"), msg="The simulation platform needs to be set to CUDA")
        self.set_active('device',None,int,"Specify the device (GPU) number; will default to the fastest available.", depend=(self.platform in ["CUDA", "OpenCL"]), 
                        msg="The simulation platform needs to be set to CUDA or OpenCL")
        self.set_active('initial_report',False,bool,"Perform one Report prior to running any dynamics.")
        self.set_active('constraints',None,str,"Specify constraints.", allowed=[None,"HBonds","AllBonds","HAngles"])
        self.constraint_obj = {None: None, "None":None,"HBonds":HBonds,"HAngles":HAngles,"AllBonds":AllBonds}[self.constraints]
        self.set_active('rigidwater',False,bool,"Add constraints to make water molecules rigid.")
        self.set_active('constraint_tolerance',1e-5,float,"Set the constraint error tolerance in the integrator (default value recommended by Peter Eastman).")
        self.set_active('serialize',None,str,"Provide a file name for writing the serialized System object.")
        self.set_active('vdw_cutoff',0.9,float,"Separate vdW cutoff distance in nanometers for AMOEBA simulation.")
        self.set_active('polarization_direct',False,bool,"Use direct polarization in AMOEBA force field")
        self.set_active('polar_eps',1e-5,float,"Polarizable dipole convergence parameter in AMOEBA force field.",
                        depend=(not self.polarization_direct),
                        msg="Polarization_direct is turned on")
        self.set_active('aewald',5.4459052,float,"Set the Ewald alpha parameter for periodic AMOEBA simulations.")
        self.set_active('pmegrid',None,list,"Set the PME grid for AMOEBA simulations.", depend=(self.nonbonded_method == "PME"),
                        msg="The nonbonded method must be set to PME.")
        self.set_active('pdb_report_interval',0,int,"Specify a timestep interval for PDB reporter.")
        self.set_active('pdb_report_filename',"output_%s.pdb" % basename,str,"Specify an file name for writing output PDB file.",
                        depend=(self.pdb_report_interval > 0), msg="pdb_report_interval needs to be set to a whole number.")
        self.set_active('dcd_report_interval',0,int,"Specify a timestep interval for DCD reporter.")
        self.set_active('dcd_report_filename',"output_%s.dcd" % basename,str,"Specify an file name for writing output DCD file.",
                        depend=(self.dcd_report_interval > 0), msg="dcd_report_interval needs to be set to a whole number.")
        self.set_active('eda_report_interval',0,int,"Specify a timestep interval for Energy reporter.", clash=(self.integrator=="mtsvvvr"), msg="EDA reporter incompatible with MTS integrator.")
        self.set_active('eda_report_filename',"output_%s.eda" % basename,str,"Specify an file name for writing output Energy file.",
                        depend=(self.eda_report_interval > 0), msg="eda_report_interval needs to be set to a whole number.")
        if self.pmegrid != None:
            assert len(self.pmegrid) == 3, "The pme argument must be a length-3 list of integers"
        self.set_active('tinkerpath',None,str,"Specify a path for TINKER executables for running AMOEBA validation.")
        if self.tinkerpath != None:
            assert os.path.exists(os.path.join(self.tinkerpath,'dynamic')), "tinkerpath must point to a directory that contains TINKER executables."

#================================#
#    The command line parser     #
#================================#

# Taken from MSMBulder - it allows for easy addition of arguments and allows "-h" for help.
def add_argument(group, *args, **kwargs):
    if 'default' in kwargs:
        d = 'Default: {d}'.format(d=kwargs['default'])
        if 'help' in kwargs:
            kwargs['help'] += ' {d}'.format(d=d)
        else:
            kwargs['help'] = d
    group.add_argument(*args, **kwargs)

print
print " #===========================================#"
print " #|    OpenMM general purpose simulation    |#"
print " #| (Hosted @ github.com/leeping/OpenMM-MD) |#"
print " #|  Use the -h argument for detailed help  |#"
print " #===========================================#"
print

parser = argparse.ArgumentParser()
add_argument(parser, 'pdb', nargs=1, metavar='input.pdb', help='Specify one PDB or AMBER inpcrd file \x1b[1;91m(Required)\x1b[0m', type=str)
add_argument(parser, 'xml', nargs='+', metavar='forcefield.xml', help='Specify multiple force field XML files, one System XML file, or one AMBER prmtop file \x1b[1;91m(Required)\x1b[0m', type=str)
add_argument(parser, '-I', '--inputfile', help='Specify an input file with options in simple two-column format.  This script will autogenerate one for you', default=None, type=str)
cmdline = parser.parse_args()
pdbfnm = cmdline.pdb[0]
xmlfnm = cmdline.xml
args = SimulationOptions(cmdline.inputfile, pdbfnm)

#================================#
#  Define custom reporters here  #
#================================#

class ProgressReport(object):
    def __init__(self, file, reportInterval, simulation, total, first=0):
        self._reportInterval = reportInterval
        self._openedFile = isinstance(file, str)
        self._initial = True
        if self._openedFile:
            self._out = open(file, 'w')
        else:
            self._out = file
        self._interval = args.report_interval * args.timestep * femtosecond
        self._units = OrderedDict()
        self._units['energy'] = kilojoule_per_mole
        self._units['kinetic'] = kilojoule_per_mole
        self._units['potential'] = kilojoule_per_mole
        self._units['temperature'] = kelvin
        if simulation.topology.getUnitCellDimensions() != None :
            self._units['density'] = kilogram / meter**3
            self._units['volume'] = nanometer**3
        self._data = defaultdict(list)
        self._total = total
        # The time step at the creation of this report.
        self._first = first
        self.run_time = 0.0*picosecond
        self.t0 = time.time()
    
    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, False, True)

    def analyze(self, simulation):
        PrintDict = OrderedDict()
        for datatype in self._units:
            data   = np.array(self._data[datatype])
            mean   = np.mean(data)
            dmean  = data - mean
            stdev  = np.std(dmean)
            g      = statisticalInefficiency(dmean)
            stderr = np.sqrt(g) * stdev / np.sqrt(len(data))
            acorr  = 0.5*(g-1)*self._interval/picosecond
            # Perform a linear fit.
            x      = np.linspace(0, 1, len(data))
            z      = np.polyfit(x, data, 1)
            p      = np.polyval(z, x)
            # Compute the drift.
            drift  = p[-1] - p[0]
            # Compute the driftless standard deviation.
            stdev1 = np.std(data-p)
            PrintDict[datatype+" (%s)" % self._units[datatype]] = "%13.5f %13.5e %13.5f %13.5f %13.5f %13.5e" % (mean, stdev, stderr, acorr, drift, stdev1)
        printcool_dictionary(PrintDict,"Summary statistics - total simulation time %.3f ps:\n%-26s %13s %13s %13s %13s %13s %13s\n%-26s %13s %13s %13s %13s %13s %13s" % (self.run_time/picosecond, 
                                                                                                                                                                          "", "", "", "", "", "", "Stdev", 
                                                                                                                                                                          "Quantity", "Mean", "Stdev", "Stderr", "Acorr(ps)", "Drift", "(NoDrift)"),keywidth=30)
    def report(self, simulation, state):
        # Compute total mass in grams.
        mass = compute_mass(simulation.system).in_units_of(gram / mole) /  AVOGADRO_CONSTANT_NA
        # The center-of-mass motion remover subtracts 3 more DoFs
        ndof = 3*simulation.system.getNumParticles() - simulation.system.getNumConstraints() - 3
        kinetic = state.getKineticEnergy()
        potential = state.getPotentialEnergy() / self._units['potential']
        kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
        temperature = 2.0 * kinetic / kB / ndof / self._units['temperature']
        kinetic /= self._units['kinetic']
        energy = kinetic + potential
        pct = 100 * float(simulation.currentStep - self._first) / self._total
        self.run_time = float(simulation.currentStep - self._first) * args.timestep * femtosecond
        if pct != 0.0:
            timeleft = (time.time()-self.t0)*(100.0 - pct)/pct
        else:
            timeleft = 0.0
        if simulation.topology.getUnitCellDimensions() != None :
            box_vectors = state.getPeriodicBoxVectors()
            volume = compute_volume(box_vectors) / self._units['volume']
            density = (mass / compute_volume(box_vectors)) / self._units['density']
            if self._initial:
                logger.info("%8s %17s %13s %13s %13s %13s %13s %13s %13s" % ('Progress', 'E.T.A', 'Time(ps)', 'Temp(K)', 'Kin(kJ)', 'Pot(kJ)', 'Ene(kJ)', 'Vol(nm3)', 'Rho(kg/m3)'))
            logger.info("%7.3f%% %17s %13.5f %13.5f %13.5f %13.5f %13.5f %13.5f %13.5f" % (pct, GetTime(timeleft), self.run_time / picoseconds, temperature, kinetic, potential, energy, volume, density))
            self._data['volume'].append(volume)
            self._data['density'].append(density)
        else:
            if self._initial:
                logger.info("%8s %17s %13s %13s %13s %13s %13s" % ('Progress', 'E.T.A', 'Time(ps)', 'Temp(K)', 'Kin(kJ)', 'Pot(kJ)', 'Ene(kJ)'))
            logger.info("%7.3f%% %17s %13.5f %13.5f %13.5f %13.5f %13.5f" % (pct, GetTime(timeleft), self.run_time / picoseconds, temperature, kinetic, potential, energy))
        self._data['energy'].append(energy)
        self._data['kinetic'].append(kinetic)
        self._data['potential'].append(potential)
        self._data['temperature'].append(temperature)
        self._initial = False
    
    def __del__(self):
        if self._openedFile:
            self._out.close()

class EnergyReporter(object):
    def __init__(self, file, reportInterval, first=0):
        self._reportInterval = reportInterval
        self._openedFile = isinstance(file, str)
        self._initial = True
        if self._openedFile:
            self._out = open(file, 'w')
        else:
            self._out = file
        # The time step at the creation of this report.
        self._first = first
    
    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation, state):
        self.run_time = float(simulation.currentStep - self._first) * args.timestep * femtosecond
        self.eda = EnergyDecomposition(simulation)
        if self._initial:
            print >> self._out, ' '.join(["%25s" % i for i in ['#Time(ps)'] + self.eda.keys()])
        print >> self._out, ' '.join(["%25.10f" % i for i in [self.run_time/picosecond] + self.eda.values()])
        self._initial = False
    
    def __del__(self):
        if self._openedFile:
            self._out.close()

class RestartReporter(object):
    def __init__(self, file, reportInterval, integrator, timestep):
        self._reportInterval = reportInterval
        self._file = file
        self._integrator = integrator
        self._timestep = timestep
    
    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation, state):
        final_state = simulation.context.getState(getEnergy=True,getPositions=True,getVelocities=True,getForces=True)
        Xfin = final_state.getPositions() / nanometer
        Vfin = final_state.getVelocities() / nanometer * picosecond
        if self._integrator not in ["velocity-verlet", "mtsvvvr"]:
            # We will attempt to get the velocities at the current time.  First obtain initial velocities.
            v0 = Vfin * nanometer / picosecond
            frc = simulation.context.getState(getForces=True).getForces()
            # Obtain masses.
            mass = []
            for i in range(simulation.context.getSystem().getNumParticles()):
                mass.append(simulation.context.getSystem().getParticleMass(i)/dalton)
            mass *= dalton
            # Get accelerations.
            accel = []
            for i in range(simulation.context.getSystem().getNumParticles()):
                accel.append(frc[i] / mass[i] / (kilojoule/(nanometer*mole*dalton)))# / (kilojoule/(nanometer*mole*dalton)))
            accel *= kilojoule/(nanometer*mole*dalton)
            # Propagate velocities backward by half a time step.
            dv = femtosecond * accel
            dv *= (+0.5 * self._timestep)
            vmdt2 = []
            for i in range(simulation.context.getSystem().getNumParticles()):
                vmdt2.append((v0[i]/(nanometer/picosecond)) + (dv[i]/(nanometer/picosecond)))
            # These are the velocities that we store (make sure it is unitless).
            Vfin = vmdt2
        Bfin = final_state.getPeriodicBoxVectors() / nanometer
        # Back up the existing file
        if os.path.exists(self._file):
            shutil.move(self._file, self._file+".bak")
        # Write the restart file pickle
        with open(self._file,'w') as f: pickle.dump((Xfin, Vfin, Bfin),f)

# Create an OpenMM PDB object.
pdb = PDBFile(pdbfnm)

# Detect the presence of periodic boundary conditions in the PDB file.
pbc = pdb.getTopology().getUnitCellDimensions() != None
if pbc:
    logger.info("Detected periodic boundary conditions")
else:
    logger.info("This is a nonperiodic simulation")


#==================================#
#|    Set up the System object.   |#
#==================================#
Deserialize = False
if len(xmlfnm) == 1 and os.path.exists(xmlfnm[0]):
    # Detect whether the input argument is an XML file or not.
    xmlroot = ET.parse(xmlfnm[0]).getroot()
    if 'type' in xmlroot.attrib and xmlroot.attrib['type'].lower() == 'system':
        Deserialize = True
        logger.info("Loading simulation settings from System XML file")
    elif xmlroot.tag.lower()  == 'forcefield':
        pass
    else:
        raise Exception('Unrecognized XML File (not OpenMM Force Field or System XML File!)')

if Deserialize:
    #===================================================================#
    #| The system XML file contains all of the settings related to the |#
    #| molecular interactions, and it can also contain thermostats and |#
    #| barostats in the form of Forces.  It is possible to override    |#
    #| some of the settings in the system XML file, and we do so here  |#
    #| for certain settings (for the sake of convenience.)             |#
    #===================================================================#
    # This line deserializes the system XML file.
    system = XmlSerializer.deserializeSystem(open(xmlfnm[0]).read())
    forces = system.getForces()
    if any(['Amoeba' in f.__class__.__name__ for f in forces]):
        logger.info("Detected AMOEBA system!")
        for f in forces:
            if f.__class__.__name__ == "AmoebaMultipoleForce":
                if 'polarization_direct' in args.UserOptions and 'polarization_direct' in args.ActiveOptions:
                    logger.info("Setting direct polarization")
                    f.setPolarizationType(1)
                    args.deactivate('polar_eps', "Not relevant for direct polarization")
                elif 'polar_eps' in args.UserOptions and 'polar_eps' in args.ActiveOptions:
                    logger.info("Setting mutual polarization with tolerance % .2e" % args.polar_eps)
                    f.setPolarizationType(0)
                    f.setMutualInducedTargetEpsilon(args.polar_eps)
                else: 
                    args.deactivate('polarization_direct', "Loaded from System XML file because not explicitly specified")
                    args.deactivate('polar_eps', "Loaded from System XML file because not explicitly specified")
        args.deactivate('vdw_cutoff', "Specified by the System XML file")
        args.deactivate('aewald', "Specified by the System XML file")
        args.deactivate('pmegrid', "Specified by the System XML file")
    else:
        args.deactivate("vdw_cutoff",msg="Not simulating an AMOEBA system")
        args.deactivate("polarization_direct",msg="Not simulating an AMOEBA system")
        args.deactivate("polar_eps",msg="Not simulating an AMOEBA system")
        args.deactivate("aewald",msg="Not simulating an AMOEBA system")
        args.deactivate("pmegrid",msg="Not simulating an AMOEBA system")
        args.deactivate("tinkerpath",msg="Not simulating an AMOEBA system")
    args.deactivate('nonbonded_method', "Specified by the System XML file")
    args.deactivate('nonbonded_cutoff', "Specified by the System XML file")
    args.deactivate('vdw_switch', "Specified by the System XML file")
    args.deactivate('switch_distance', "Specified by the System XML file")
    args.deactivate('constraints', "Specified by the System XML file")
    args.deactivate('rigidwater', "Specified by the System XML file")
    args.deactivate('ewald_error_tolerance', "Specified by the System XML file")
    args.deactivate('dispersion_correction', "Specified by the System XML file")
else:
    settings = []
    # This creates a system from a force field XML file.
    forcefield = ForceField(*xmlfnm)
    if any(['Amoeba' in f.__class__.__name__ for f in forcefield._forces]):
        logger.info("Detected AMOEBA system!")
        settings += [('constraints', args.constraints), ('rigidWater', args.rigidwater)]
        if pbc:
            logger.info("Periodic AMOEBA system uses PME regardless of user-supplied options.")
            args.force_active('nonbonded_method',"PME","PME enforced for periodic AMOEBA system.")
            settings += [('nonbondedMethod', PME), ('nonbondedCutoff', args.nonbonded_cutoff * nanometer), 
                         ('vdwCutoff', args.vdw_cutoff), ('useDispersionCorrection', args.dispersion_correction)]
            if 'pmegrid' in args.UserOptions and 'aewald' in args.UserOptions:
                settings.append(('pmeGridDimensions', args.pmegrid))
                settings.append(('aEwald', args.aewald))
                args.deactivate('ewald_error_tolerance',"PME grid was explicitly specified")
                if 'ewald_error_tolerance' in args.UserOptions:
                    logger.info("Both pmegrid and aEwald have been set, ewald_error_tolerance is not used.")
            elif 'ewald_error_tolerance' in args.UserOptions:
                settings.append(('ewaldErrorTolerance', args.ewald_error_tolerance))
                args.deactivate('pmegrid',"pmegrid and aewald must both be specified if they are to be used")
                args.deactivate('aewald',"pmegrid and aewald must both be specified if they are to be used")
            args.deactivate('vdw_switch', "AMOEBA vdW interaction has switch function by default.")
            args.deactivate('switch_distance', "AMOEBA vdW interaction has switch function by default.")
        else:
            args.force_active('nonbonded_method',"NoCutoff","Nonbonded method forced to NoCutoff for nonperiodic AMOEBA system.")
            args.deactivate('nonbonded_cutoff',"Deactivated because nonbonded method forced to NoCutoff")
            args.deactivate('aewald',"Deactivated because nonbonded method forced to NoCutoff")
            args.deactivate('ewald_error_tolerance',"Deactivated because nonbonded method forced to NoCutoff")
            args.deactivate('vdw_cutoff',"Deactivated because nonbonded method forced to NoCutoff")
            args.deactivate('dispersion_correction',"Deactivated because nonbonded method forced to NoCutoff")
            args.deactivate("pmegrid",msg="Deactivated because nonbonded method forced to NoCutoff")
            args.deactivate('vdw_switch', "Deactivated because nonbonded method forced to NoCutoff")
            args.deactivate('switch_distance', "Deactivated because nonbonded method forced to NoCutoff")
            settings.append(('nonbondedMethod', NoCutoff))
        if args.polarization_direct:
            logger.info("Setting direct polarization")
            args.deactivate('polar_eps', "Not relevant for direct polarization")
            settings.append(('polarization', 'direct'))
        else:
            logger.info("Setting mutual polarization with tolerance % .2e" % args.polar_eps)
            settings.append(('mutualInducedTargetEpsilon', args.polar_eps))
    else:
        logger.info("Detected non-AMOEBA system")
        if pbc:
            settings = [('constraints', args.constraints), ('rigidWater', args.rigidwater), ('nonbondedMethod', args.nonbonded_method_obj), 
                        ('nonbondedCutoff', args.nonbonded_cutoff * nanometer), ('useDispersionCorrection', True)]
            if (args.nonbonded_method_obj in [Ewald, PME]) and ('ewald_error_tolerance' in args.UserOptions):
                settings += [('ewaldErrorTolerance', args.ewald_error_tolerance)]
        else:
            if args.nonbonded_method_obj in [CutoffPeriodic, Ewald, PME]:
                raise Exception('Nonbonded methods CutoffPeriodic, Ewald, or PME cannot be used for nonperiodic systems; use NoCutoff or CutoffNonPeriodic instead.')
            if args.nonbonded_method_obj == NoCutoff:
                args.deactivate('nonbonded_cutoff',"Not using cutoffs for nonbonded interactions.")
                args.deactivate('vdw_switch', "Deactivated because nonbonded method forced to NoCutoff")
                args.deactivate('switch_distance', "Deactivated because nonbonded method forced to NoCutoff")
            settings = [('constraints', args.constraints), ('rigidWater', args.rigidwater), ('nonbondedMethod', args.nonbonded_method_obj)]
            args.deactivate('ewald_error_tolerance',"Deactivated for a nonperiodic system.")
            args.deactivate('dispersion_correction',"Deactivated for a nonperiodic system.")
        # if 'vdw_switch' in args.ActiveOptions and args.vdw_switch:
        #     settings += [('useSwitchingFunction', True), ('switchingDistance', args.switch_distance)]
        args.deactivate("vdw_cutoff",msg="Not simulating an AMOEBA system")
        args.deactivate("polarization_direct",msg="Not simulating an AMOEBA system")
        args.deactivate("polar_eps",msg="Not simulating an AMOEBA system")
        args.deactivate("aewald",msg="Not simulating an AMOEBA system")
        args.deactivate("pmegrid",msg="Not simulating an AMOEBA system")
        args.deactivate("tinkerpath",msg="Not simulating an AMOEBA system")
    logger.info("Now setting up the System with the following system settings:")
    printcool_dictionary(dict(settings),title="OpenMM system object will be set up\n using these options:")
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addExtraParticles(forcefield)
    system = forcefield.createSystem(modeller.topology, **dict(settings))

#====================================#
#| Temperature and pressure control |#
#====================================#

sysxml_thermo = False
sysxml_baro = False
if Deserialize:
    for f in forces:
        if f.__class__.__name__ == "MonteCarloBarostat":
            sysxml_baro = True
            args.deactivate("pressure", msg="Specified by the System XML file")
            args.deactivate("nbarostat", msg="Specified by the System XML file")
            args.deactivate("anisotropic", msg="Specified by the System XML file")
            logger.info("The system XML file contains a Monte Carlo barostat at %.2f atm pressure" % (f.getDefaultPressure()/atmosphere))
            if not pbc:
                raise Exception('System contains a Barostat but the topology contains no periodic box! Exiting...')
        if f.__class__.__name__ == "AndersenThermostat":
            sysxml_thermo = True
            args.deactivate("temperature", msg="Specified by the System XML file")
            args.deactivate("collision_rate", msg="Specified by the System XML file")
            logger.info("The system XML file contains an Andersen thermostat at %.2f K temperature" % (f.getDefaultTemperature()/kelvin))
            if args.integrator in ["langevin", "mtsvvvr"]:
                raise Exception('Cannot create a Langevin or MTS-VVVR integrator with existing temperature control! Exiting...')

def add_barostat():
    if sysxml_baro:
        logger.info("Ignoring user-specified pressure control and going with the existing barostat.")
    else:
        if args.pressure <= 0.0:
            logger.info("This is a constant volume (NVT) run")
        elif pbc:
            logger.info("This is a constant pressure (NPT) run at %.2f atm pressure" % args.pressure)
            logger.info("Adding Monte Carlo barostat with volume adjustment interval %i" % args.nbarostat)
            logger.info("Anisotropic box scaling is %s" % ("ON" if args.anisotropic else "OFF"))
            if args.anisotropic:
                logger.info("Only the Z-axis will be adjusted")
                barostat = MonteCarloAnisotropicBarostat(Vec3(args.pressure*atmospheres, args.pressure*atmospheres, args.pressure*atmospheres), args.temperature*kelvin, False, False, True, args.nbarostat)
            else:
                barostat = MonteCarloBarostat(args.pressure * atmospheres, args.temperature * kelvin, args.nbarostat)
            system.addForce(barostat)
        else:
            args.deactivate("pressure", msg="System is nonperiodic")
            #raise Exception('Pressure was specified but the topology contains no periodic box! Exiting...')

def VelocityVerletIntegrator(timestep):
    # Velocity Verlet integrator with explicit velocities.
    integrator = CustomIntegrator(timestep/picosecond)
    integrator.addPerDofVariable("x1", 0)
    integrator.addPerDofVariable("x2", 0)
    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    integrator.addComputePerDof("x", "x+dt*v")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions()
    integrator.addComputePerDof("x2", "x")
    integrator.addComputePerDof("v", "v+0.5*dt*f/m+(x2-x1)/dt")
    integrator.addConstrainVelocities()
    return integrator

def NVEIntegrator():
    if args.integrator == "verlet":
        logger.info("Creating a Leapfrog integrator with %.2f fs timestep." % args.timestep)
        integrator = VerletIntegrator(args.timestep * femtosecond)
    elif args.integrator == "velocity-verlet":
        logger.info("Creating a Velocity Verlet integrator with %.2f fs timestep." % args.timestep)
        integrator = VelocityVerletIntegrator(args.timestep * femtosecond)
    return integrator

if sysxml_thermo:
    logger.info("Ignoring user-specified temperature control.")
    integrator = NVEIntegrator()
    add_barostat()
else:
    if args.temperature <= 0.0:
        logger.info("This is a constant energy, constant volume (NVE) run.")
        integrator = NVEIntegrator()
    else:
        logger.info("This is a constant temperature run at %.2f K" % args.temperature)
        logger.info("The stochastic thermostat collision frequency is %.2f ps^-1" % args.collision_rate)
        if args.integrator == "langevin":
            logger.info("Creating a Langevin integrator with %.2f fs timestep." % args.timestep)
            integrator = LangevinIntegrator(args.temperature * kelvin, args.collision_rate / picosecond, args.timestep * femtosecond)
        elif args.integrator == "mtsvvvr":
            logger.info("Creating a multiple timestep Langevin integrator with %.2f / %.2f fs outer/inner timestep." % (args.timestep, args.innerstep))
            if int(args.timestep / args.innerstep) != args.timestep / args.innerstep:
                raise Exception("The inner step must be an even subdivision of the time step.")
            integrator = MTSVVVRIntegrator(args.temperature * kelvin, args.collision_rate / picosecond, args.timestep * femtosecond, system, int(args.timestep / args.innerstep))
        else:
            integrator = NVEIntegrator()
            thermostat = AndersenThermostat(args.temperature * kelvin, args.collision_rate / picosecond)
            system.addForce(thermostat)
        if sysxml_baro:
            logger.info("Ignoring user-specified pressure control and going with the existing barostat.")
        else:
            add_barostat()

if not hasattr(args,'constraints') or (str(args.constraints) == "None" and args.rigidwater == False):
    args.deactivate('constraint_tolerance',"There are no constraints in this system")
else:
    integrator.setConstraintTolerance(args.constraint_tolerance)

#==================================#
#|      Create the platform       |#
#==================================#
# if args.platform != None:
logger.info("Setting Platform to %s" % str(args.platform))
try:
    platform = Platform.getPlatformByName(args.platform)
except:
    logger.info("Warning: %s platform not found, going to Reference platform \x1b[91m(slow)\x1b[0m" % args.platform)
    args.force_active('platform',"Reference","The %s platform was not found." % args.platform)
    platform = Platform.getPlatformByName("Reference")

if 'device' in args.ActiveOptions:
    # The device may be set using an environment variable or the input file.
    if os.environ.has_key('CUDA_DEVICE'):
        device = os.environ.get('CUDA_DEVICE',str(args.device))
    elif os.environ.has_key('CUDA_DEVICE_INDEX'):
        device = os.environ.get('CUDA_DEVICE_INDEX',str(args.device))
    else:
        device = str(args.device)
    if device != None:
        logger.info("Setting Device to %s" % str(device))
        #platform.setPropertyDefaultValue("CudaDevice", device)
        platform.setPropertyDefaultValue("CudaDeviceIndex", device)
        #platform.setPropertyDefaultValue("OpenCLDeviceIndex", device)
    else:
        logger.info("Using the default (fastest) device")
else:
    logger.info("Using the default (fastest) device")
if "CudaPrecision" in platform.getPropertyNames():
    platform.setPropertyDefaultValue("CudaPrecision", args.cuda_precision)
# else:
#     logger.info("Using the default Platform")

#==================================#
#|  Create the simulation object  |#
#==================================#
logger.info("Creating the Simulation object")
# Get the number of forces and set each force to a different force group number.
nfrc = system.getNumForces()
if args.integrator != 'mtsvvvr':
    for i in range(nfrc):
        system.getForce(i).setForceGroup(i)
for i in range(nfrc):
    # Set vdW switching function manually.
    f = system.getForce(i)
    if f.__class__.__name__ == 'NonbondedForce':
        if 'vdw_switch' in args.ActiveOptions and args.vdw_switch:
            f.setUseSwitchingFunction(True)
            f.setSwitchingDistance(args.switch_distance)
if args.platform != None:
    simulation = Simulation(modeller.topology, system, integrator, platform)
else:
    simulation = Simulation(modeller.topology, system, integrator)
# Serialize the system if we want.
if args.serialize != 'None' and args.serialize != None:
    logger.info("Serializing the system")
    serial = XmlSerializer.serializeSystem(system)
    bak(args.serialize)
    with open(args.serialize,'w') as f: f.write(serial)
# Print out the platform used by the context
printcool_dictionary({i:simulation.context.getPlatform().getPropertyValue(simulation.context,i) for i in simulation.context.getPlatform().getPropertyNames()},title="Platform %s has properties:" % simulation.context.getPlatform().getName())

# Print out some more information about the system
logger.info("--== System Information ==--")
logger.info("Number of particles   : %i" % simulation.context.getSystem().getNumParticles())
logger.info("Number of constraints : %i" % simulation.context.getSystem().getNumConstraints())
logger.info("Total system mass     : %.2f amu" % (compute_mass(system)/amu))
for f in simulation.context.getSystem().getForces():
    if f.__class__.__name__ == 'AmoebaMultipoleForce':
        logger.info("AMOEBA PME order      : %i" % f.getPmeBSplineOrder())
        logger.info("AMOEBA PME grid       : %s" % str(f.getPmeGridDimensions()))
    if f.__class__.__name__ == 'NonbondedForce':
        method_names = ["NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "Ewald", "PME"]
        logger.info("Nonbonded method      : %s" % method_names[f.getNonbondedMethod()])
        logger.info("Number of particles   : %i" % f.getNumParticles())
        logger.info("Number of exceptions  : %i" % f.getNumExceptions())
        if f.getNonbondedMethod() > 0:
            logger.info("Nonbonded cutoff      : %.3f nm" % (f.getCutoffDistance() / nanometer))
            if f.getNonbondedMethod() >= 3:
                logger.info("Ewald error tolerance : %.3e" % (f.getEwaldErrorTolerance()))
            logger.info("LJ switching function : %i" % f.getUseSwitchingFunction())
            if f.getUseSwitchingFunction():
                logger.info("LJ switching distance : %.3f nm" % (f.getSwitchingDistance() / nanometer))

# Print the sample input file here.
for line in args.record():
    print line

#===============================================================#
#| Run dynamics for equilibration, or load restart information |#
#===============================================================#
if os.path.exists(args.restart_filename) and args.read_restart:
    print "Restarting simulation from the restart file."
    # Load information from the restart file.
    r_positions, r_velocities, r_boxes = pickle.load(open(args.restart_filename))
    # NOTE: Periodic box vectors must be set FIRST
    if pbc:
        simulation.context.setPeriodicBoxVectors(r_boxes[0] * nanometer,r_boxes[1] * nanometer, r_boxes[2] * nanometer)
    simulation.context.setPositions(r_positions * nanometer)
    if args.integrator not in ["velocity-verlet", "mtsvvvr"]:
        # We will attempt to reconstruct the leapfrog velocities.  First obtain initial velocities.
        v0 = r_velocities * nanometer / picosecond
        frc = simulation.context.getState(getForces=True).getForces()
        # Obtain masses.
        mass = []
        for i in range(simulation.context.getSystem().getNumParticles()):
            mass.append(simulation.context.getSystem().getParticleMass(i)/dalton)
        mass *= dalton
        # Get accelerations.
        accel = []
        for i in range(simulation.context.getSystem().getNumParticles()):
            accel.append(frc[i] / mass[i] / (kilojoule/(nanometer*mole*dalton)))# / (kilojoule/(nanometer*mole*dalton)))
        accel *= kilojoule/(nanometer*mole*dalton)
        # Propagate velocities backward by half a time step.
        dv = femtosecond * accel
        dv *= (-0.5 * args.timestep)
        vmdt2 = []
        for i in range(simulation.context.getSystem().getNumParticles()):
            vmdt2.append((v0[i]/(nanometer/picosecond)) + (dv[i]/(nanometer/picosecond)))
        vmdt2 *= nanometer/picosecond
        # Assign velocities.
        simulation.context.setVelocities(vmdt2)
        simulation.context.applyVelocityConstraints(0.0001)
    else:
        simulation.context.setVelocities(r_velocities * nanometer / picosecond)
    first = 0
else:
    # Set initial positions.
    simulation.context.setPositions(modeller.positions)
    print "Initial potential is:", simulation.context.getState(getEnergy=True).getPotentialEnergy()
    if args.integrator != 'mtsvvvr':
        eda = EnergyDecomposition(simulation)
        eda_kcal = OrderedDict([(i, "%10.4f" % (j/4.184)) for i, j in eda.items()])
        printcool_dictionary(eda_kcal, title="Energy Decomposition (kcal/mol)")
    
    # Minimize the energy.
    if args.minimize:
        print "Minimization start, the energy is:", simulation.context.getState(getEnergy=True).getPotentialEnergy()
        simulation.minimizeEnergy()
        print "Minimization done, the energy is", simulation.context.getState(getEnergy=True).getPotentialEnergy()
        positions = simulation.context.getState(getPositions=True).getPositions()
        print "Minimized geometry is written to 'minimized.pdb'"
        PDBFile.writeModel(modeller.topology, positions, open('minimized.pdb','w'))
    # Assign velocities.
    if args.gentemp > 0.0:
        logger.info("Generating velocities corresponding to Maxwell distribution at %.2f K" % args.gentemp)
        simulation.context.setVelocitiesToTemperature(args.gentemp * kelvin)
    # Equilibrate.
    logger.info("--== Equilibrating (%i steps, %.2f ps) ==--" % (args.equilibrate, args.equilibrate * args.timestep * femtosecond / picosecond))
    if args.report_interval > 0:
        # Append the ProgressReport for equilibration run.
        simulation.reporters.append(ProgressReport(sys.stdout, args.report_interval, simulation, args.equilibrate))
        logger.info("Progress will be reported every %i steps" % args.report_interval)
    # This command actually does all of the computation.
    simulation.step(args.equilibrate)
    if args.report_interval > 0:
        # Get rid of the ProgressReport because we'll make a new one.
        simulation.reporters.pop()
    first = args.equilibrate

#============================#
#| Production MD simulation |#
#============================#
logger.info("--== Production (%i steps, %.2f ps) ==--" % (args.production, args.production * args.timestep * femtosecond / picosecond))

#===========================================#
#| Add reporters for production simulation |#
#===========================================#
if args.report_interval > 0:
    logger.info("Progress will be reported every %i steps" % args.report_interval)
    simulation.reporters.append(ProgressReport(sys.stdout, args.report_interval, simulation, args.production, first))

if args.pdb_report_interval > 0:
    bak(args.pdb_report_filename)
    logger.info("PDB Reporter will write to %s every %i steps" % (args.pdb_report_filename, args.pdb_report_interval))
    simulation.reporters.append(PDBReporter(args.pdb_report_filename, args.pdb_report_interval))

if args.dcd_report_interval > 0:
    bak(args.dcd_report_filename)
    logger.info("DCD Reporter will write to %s every %i steps" % (args.dcd_report_filename, args.dcd_report_interval))
    simulation.reporters.append(DCDReporter(args.dcd_report_filename, args.dcd_report_interval))

if args.restart_interval > 0:
    bak(args.restart_filename)
    logger.info("Restart information will be written to %s every %i steps" % (args.restart_filename, args.restart_interval))
    simulation.reporters.append(RestartReporter(args.restart_filename, args.restart_interval, args.integrator, args.timestep))

if 'eda_report_interval' in args.ActiveOptions and args.eda_report_interval > 0:
    bak(args.eda_report_filename)
    logger.info("Energy Reporter will write to %s every %i steps" % (args.eda_report_filename, args.eda_report_interval))
    simulation.reporters.append(EnergyReporter(args.eda_report_filename, args.eda_report_interval, first))

if hasattr(args,'tinkerpath') and args.tinkerpath != None:
    logger.info("Appending the TINKER Reporter.")
    simulation.reporters.append(TinkerReporter(sys.stdout, nsteps, simulation, os.path.splitext(pdbfnm)[0]+".xyz", pbc=pbc, pmegrid=args.pmegrid, tinkerpath=args.tinkerpath))

# Do one report before running any dynamics.
if args.initial_report:
    logger.info("Doing the initial report.")
    for Reporter in simulation.reporters:
        #if Reporter.__class__.__name__ != "ProgressReport":
        Reporter.report(simulation,simulation.context.getState(getPositions=True,getVelocities=True,getForces=True,getEnergy=True))

t1 = time.time()

# This command actually does all of the computation.
simulation.step(args.production)

prodtime = time.time() - t1

#=============================================#
#| Run analysis and save restart information |#
#=============================================#
logger.info('Getting statistics for the production run.')
simulation.reporters[0].analyze(simulation)
print "Total wall time: % .4f seconds" % (time.time() - t0)
print "Production wall time: % .4f seconds" % (prodtime)
print "Simulation speed: % .6f ns/day" % (86400*args.production*args.timestep*femtosecond/nanosecond/(prodtime))
