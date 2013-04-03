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
    if d.day-1 > 0:
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
            # if val != None and clash:
            #     raise Exception("Tried to set option \x1b[1;91m%s\x1b[0m to \x1b[94m%s\x1b[0m but there was a clash %s" % (key, str(val), "; \x1b[92m%s\x1b[0m" % cmsg if cmsg != None else ""))
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
            if msg != None:
                self.InactiveWarnings[key] = msg

    def deactivate(self,key,msg=None):
        """ Deactivate one option.  The arguments are:
        key     : The name of the option.
        msg     : A warning that is printed out if the option is not activated.
        """
        if key in self.ActiveOptions:
            self.InactiveOptions[key] = self.ActiveOptions[key]
            del self.ActiveOptions[key]
        if msg != None:
            self.InactiveWarnings[key] = msg
        
    def __getattr__(self,key):
        if key in self.ActiveOptions:
            return self.ActiveOptions[key]
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
        UserSupplied = []
        for key in self.ActiveOptions:
            if key in self.UserOptions:
                UserSupplied.append("%-22s %20s # %s" % (key, str(self.ActiveOptions[key]), self.Documentation[key]))
        if len(UserSupplied) > 0:
            out.append("")
            out.append("#===========================================#")
            out.append("#|          User-supplied options:         |#")
            out.append("#===========================================#")
            out += UserSupplied
        ActiveDefault = []
        for key in self.ActiveOptions:
            if key not in self.UserOptions:
                ActiveDefault.append("%-22s %20s # %s" % (key, str(self.ActiveOptions[key]), self.Documentation[key]))
        if len(ActiveDefault) > 0:
            out.append("")
            out.append("#===========================================#")
            out.append("#|   Active options at default values:     |#")
            out.append("#===========================================#")
            out += ActiveDefault
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
            out.append("")
            out.append("#===========================================#")
            out.append("#|          Unrecognized options:          |#")
            out.append("#===========================================#")
            out += Unrecognized
        out.append("")
        out.append("#===========================================#")
        out.append("#|           End of Input File             |#")
        out.append("#===========================================#")
        return out
        
    def __init__(self, input_file, pdbfnm):
        super(SimulationOptions,self).__init__()
        basename = os.path.splitext(pdbfnm)[0]
        self.Documentation = OrderedDict()
        self.UserOptions = OrderedDict()
        self.ActiveOptions = OrderedDict()
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
        self.set_active('integrator','verlet',str,"Molecular dynamics integrator",allowed=["verlet","langevin"])
        self.set_active('minimize',False,bool,"Specify whether to minimize the energy before running dynamics.")
        self.set_active('timestep',1.0,float,"Time step in femtoseconds.")
        self.set_active('restart_filename','restart.p',str,"Restart information will be read from / written to this file (will be backed up).")
        self.set_active('restart',True,bool,"Restart simulation from the restart file.",
                        depend=(os.path.exists(self.restart_filename)), msg="Cannot restart; file specified by restart_filename does not exist.")
        self.set_active('equilibrate',0,int,"Number of steps reserved for equilibration.")
        self.set_active('production',1000,int,"Number of steps in production run.")
        self.set_active('report_interval',100,int,"Number of steps between every progress report.")
        self.set_active('temperature',0.0,float,"Simulation temperature for Langevin integrator or Andersen thermostat.")
        if self.temperature <= 0.0 and self.integrator == "langevin":
            raise Exception("You need to set a finite temperature if using the Langevin integrator!")
        self.set_active('gentemp',self.temperature,float,"Specify temperature for generating velocities")
        self.set_active('collision_rate',0.1,float,"Collision frequency for Langevin integrator or Andersen thermostat in ps^-1.",
                        depend=(self.integrator == "langevin" or self.temperature != 0.0),
                        msg="We're not running a constant temperature simulation")
        self.set_active('polarization_direct',False,bool,"Use direct polarization in AMOEBA force field")
        self.set_active('polar_eps',1e-5,float,"Polarizable dipole convergence parameter in AMOEBA force field.",
                        depend=(not self.polarization_direct),
                        msg="Polarization_direct is turned on")
        self.set_active('pressure',0.0,float,"Simulation pressure; set a positive number to activate.",
                        clash=(self.temperature <= 0.0),
                        msg="For constant pressure simulations, the temperature must be finite")
        self.set_active('nbarostat',25,int,"Step interval for MC barostat volume adjustments.",
                        depend=("pressure" in self.ActiveOptions and self.pressure > 0.0), msg = "We're not running a constant pressure simulation")
        self.set_active('nonbonded_cutoff',0.9,float,"Nonbonded cutoff distance in nanometers.")
        self.set_active('vdw_cutoff',0.9,float,"Separate vdW cutoff distance in nanometers for AMOEBA simulation.")
        self.set_active('platform',None,str,"The simulation platform.", allowed=[None, "Reference","CUDA","OpenCL"])
        self.set_active('cuda_precision','single',str,"The precision of the CUDA platform.", allowed=["single","mixed","double"], 
                        depend=(self.platform == "CUDA"), msg="The simulation platform needs to be set to CUDA")
        self.set_active('device',0,int,"Specify the device (GPU) number.", depend=(self.platform in ["CUDA", "OpenCL"]), 
                        msg="The simulation platform needs to be set to CUDA or OpenCL")
        self.set_active('initial_report',False,bool,"Perform one Report prior to running any dynamics.")
        self.set_active('constraints',None,str,"Specify constraints.", allowed=[None,"HBonds","AllBonds","HAngles"])
        if type(self.constraints) == str:
            self.constraints = {"None":None,"HBonds":HBonds,"HAngles":HAngles,"AllBonds":AllBonds}[self.constraints]
        self.set_active('rigidwater',False,bool,"Make water molecules rigid.")
        self.set_active('serialize',None,str,"Provide a file name for writing the serialized System object.")
        self.set_active('pmegrid',None,list,"Set the PME grid for AMOEBA simulations.")
        self.set_active('pdb_report_interval',0,int,"Specify a timestep interval for PDB reporter.")
        self.set_active('pdb_report_filename',"output_%s.pdb" % basename,str,"Specify an file name for writing output PDB file.",
                        depend=(self.pdb_report_interval > 0), msg="pdb_report_interval needs to be set to a whole number.")
        self.set_active('dcd_report_interval',0,int,"Specify a timestep interval for DCD reporter.")
        self.set_active('dcd_report_filename',"output_%s.dcd" % basename,str,"Specify an file name for writing output DCD file.",
                        depend=(self.dcd_report_interval > 0), msg="dcd_report_interval needs to be set to a whole number.")
        self.set_active('eda_report_interval',0,int,"Specify a timestep interval for Energy reporter.")
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
print " #=========================================#"
print " #|   OpenMM general purpose simulation   |#"
print " #| Use the -h argument for detailed help |#"
print " #=========================================#"
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
        return (steps, True, True, True, True)

    def analyze(self, simulation):
        PrintDict = OrderedDict()
        for datatype in self._units:
            data   = np.array(self._data[datatype])
            mean   = np.mean(data)
            stdev  = np.std(data)
            g      = statisticalInefficiency(data)
            stderr = np.sqrt(g) * stdev / np.sqrt(len(data))
            acorr  = 0.5*(g-1)*self._interval/picosecond
            PrintDict[datatype+" (%s)" % self._units[datatype]] = "%12.3f %12.3f %12.3f %12.3f" % (mean, stdev, stderr, acorr)
        printcool_dictionary(PrintDict,"Summary statistics - total simulation time %.3f ps:\n%-26s %12s %12s %12s %12s" % (self.run_time/picosecond, "Quantity", "Mean", 
                                                                                                                           "Stdev", "Stderr", "Acorr(ps)"),keywidth=30)
    def report(self, simulation, state):
        # Compute total mass in grams.
        mass = compute_mass(simulation.system).in_units_of(gram / mole) /  AVOGADRO_CONSTANT_NA
        # The center-of-mass motion remover subtracts 3 more DoFs
        ndof = 3*simulation.system.getNumParticles() - simulation.system.getNumConstraints() - 3
        kinetic = state.getKineticEnergy()
        potential = state.getPotentialEnergy() / self._units['potential']
        kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
        temperature = 2.0 * kinetic / kB / ndof / self._units['temperature']
        pct = 100 * float(simulation.currentStep - self._first) / self._total
        self.run_time = float(simulation.currentStep - self._first) * args.timestep * femtosecond
        timeleft = (time.time()-self.t0)*(100.0 - pct)/pct

        if simulation.topology.getUnitCellDimensions() != None :
            box_vectors = state.getPeriodicBoxVectors()
            volume = compute_volume(box_vectors) / self._units['volume']
            density = (mass / compute_volume(box_vectors)) / self._units['density']
            if self._initial:
                logger.info("%8s %12s %9s %9s %13s %10s %13s" % ('Progress', 'E.T.A', 'Time(ps)', 'Temp(K)', 'Pot(kJ)', 'Vol(nm3)', 'Rho(kg/m3)'))
            logger.info("%7.3f%% %12s %9.3f %9.3f % 13.3f %10.4f %13.4f" % (pct, GetTime(timeleft), self.run_time / picoseconds, temperature, potential, volume, density))
            self._data['volume'].append(volume)
            self._data['density'].append(density)
        else:
            if self._initial:
                logger.info("%8s %12s %9s %9s %9s %13s" % ('Progress', 'E.T.A', 'Time(ps)', 'Temp(K)', 'Pot(kJ)'))
            logger.info("%7.3f%% %12s %8is %9.3f %9.3f % 13.3f" % (pct, GetTime(timeleft), self.run_time / picoseconds, temperature, potential))
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
        return (steps, True, True, True, True)

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
if len(xmlfnm[0]) == 1 and os.path.exists(xmlfnm[0]):
    # Detect whether the input argument is an XML file or not.
    xmlroot = ET.parse(xmlfnm[0]).getroot()
    if 'type' in xmlroot.attrib and xmlroot.attrib['type'].lower() == 'system':
        Deserialize = True
        logger.info("Loading all simulation settings from System XML file ; user-provided settings are ignored!")
    else:
        raise Exception('Unrecognized XML File (not OpenMM Force Field or System XML File!)')

if Deserialize:
    # This line would deserialize a system XML file.
    system = XmlSerializer.deserializeSystem(open(xmlfnm[0]).read())
else:
    settings = []
    # This creates a system from a force field XML file.
    forcefield = ForceField(*xmlfnm)
    if any(['Amoeba' in i.__class__.__name__ for i in forcefield._forces]):
        logger.info("Detected AMOEBA system!")
        settings += [('constraints', args.constraints), ('rigidWater', args.rigidwater)]
        if pbc:
            settings += [('nonbondedMethod', PME), ('nonbondedCutoff', args.nonbonded_cutoff * nanometer), 
                         ('vdwCutoff', args.vdw_cutoff), ('aEwald', 5.4459052), ('useDispersionCorrection', True)]
            if args.pmegrid != None:
                settings.append(('pmeGridDimensions', args.pmegrid))
        if args.polarization_direct:
            logger.info("Setting direct polarization")
            settings.append(('polarization', 'direct'))
        else:
            logger.info("Setting mutual polarization")
            settings.append(('mutualInducedTargetEpsilon', args.polar_eps))
    else:
        logger.info("Detected non-AMOEBA system")
        if pbc:
            settings = [('constraints', args.constraints), ('rigidWater', args.rigidwater), ('nonbondedMethod', PME), 
                        ('nonbondedCutoff', args.nonbonded_cutoff * nanometer), ('useDispersionCorrection', True)]
        else:
            settings = [('constraints', args.constraints), ('rigidWater', args.rigidwater), ('nonbondedMethod', NoCutoff)]
        args.deactivate("polarization_direct",msg="Not simulating an AMOEBA system")
        args.deactivate("polar_eps",msg="Not simulating an AMOEBA system")
        args.deactivate("vdw_cutoff",msg="Not simulating an AMOEBA system")
        args.deactivate("pmegrid",msg="Not simulating an AMOEBA system")
        args.deactivate("tinkerpath",msg="Not simulating an AMOEBA system")
    logger.info("Now setting up the System")
    system = forcefield.createSystem(pdb.topology, **dict(settings))

logger.info("--== System Information ==--")
logger.info("Number of particles   : %i" % system.getNumParticles())
logger.info("Number of constraints : %i" % system.getNumConstraints())
logger.info("Total system mass     : %.2f amu" % (compute_mass(system)/amu))

#====================================#
#| Temperature and pressure control |#
#====================================#
if args.temperature <= 0.0:
    logger.info("This is a constant energy run using the Verlet integrator.")
    integrator = VerletIntegrator(args.timestep * femtosecond)
else:
    logger.info("This is a constant temperature run at %.2f K" % args.temperature)
    logger.info("The stochastic thermostat collision frequency is %.2f ps^-1" % args.collision_rate)
    if args.integrator == "langevin":
        logger.info("Creating a Langevin integrator")
        integrator = LangevinIntegrator(args.temperature * kelvin, args.collision_rate / picosecond, args.timestep * femtosecond)
    else:
        logger.info("Using Verlet integrator with Andersen thermostat")
        integrator = VerletIntegrator(args.timestep * femtosecond)
        thermostat = AndersenThermostat(args.temperature * kelvin, args.collision_rate / picosecond)
        system.addForce(thermostat)
    if args.pressure <= 0.0:
        logger.info("This is a constant volume run")
    elif pbc:
        logger.info("This is a constant pressure run at %.2f atm pressure" % args.pressure)
        logger.info("Adding Monte Carlo barostat with volume adjustment interval %i" % args.nbarostat)
        barostat = MonteCarloBarostat(args.pressure * atmospheres, args.temperature * kelvin, args.nbarostat)
        system.addForce(barostat)
    else:
        raise Exception('Pressure was specified but the system is nonperiodic! Exiting...')

#==================================#
#|      Create the platform       |#
#==================================#
if args.platform != None:
    logger.info("Setting Platform to %s" % str(args.platform))
    platform = Platform.getPlatformByName(args.platform)
    if 'device' in args.ActiveOptions:
        # The device may be set using an environment variable or the input file.
        if os.environ.has_key('CUDA_DEVICE'):
            device = os.environ.get('CUDA_DEVICE',str(args.device))
        elif os.environ.has_key('CUDA_DEVICE_INDEX'):
            device = os.environ.get('CUDA_DEVICE_INDEX',str(args.device))
        else:
            device = str(args.device)
        logger.info("Setting Device to %s" % str(device))
        platform.setPropertyDefaultValue("CudaDevice", device)
        platform.setPropertyDefaultValue("CudaDeviceIndex", device)
        platform.setPropertyDefaultValue("OpenCLDeviceIndex", device)
        platform.setPropertyDefaultValue("CudaPrecision", args.cuda_precision)
else:
    print "Using the default Platform"

#==================================#
#|  Create the simulation object  |#
#==================================#
logger.info("Creating the Simulation object")
# Get the number of forces and set each force to a different force group number.
nfrc = system.getNumForces()
for i in range(nfrc):
    system.getForce(i).setForceGroup(i)
if args.platform != None:
    simulation = Simulation(pdb.topology, system, integrator, platform)
else:
    simulation = Simulation(pdb.topology, system, integrator)
# Serialize the system if we want.
if args.serialize != 'None' and args.serialize != None:
    logger.info("Serializing the system")
    serial = XmlSerializer.serializeSystem(system)
    bak(args.serialize)
    with open(args.serialize,'w') as f: f.write(serial)
# Print out the platform used by the context
printcool_dictionary({i:simulation.context.getPlatform().getPropertyValue(simulation.context,i) for i in simulation.context.getPlatform().getPropertyNames()},title="Platform %s has properties:" % simulation.context.getPlatform().getName())

# Print the sample input file here.
for line in args.record():
    print line

#===============================================================#
#| Run dynamics for equilibration, or load restart information |#
#===============================================================#
if os.path.exists(args.restart_filename) and args.restart:
    print "Restarting simulation from the restart file."
    # Load information from the restart file.
    r_positions, r_velocities, r_boxes = pickle.load(open(args.restart_filename))
    simulation.context.setPositions(r_positions * nanometer)
    simulation.context.setVelocities(r_velocities * nanometer / picosecond)
    if pbc:
        simulation.context.setPeriodicBoxVectors(r_boxes[0] * nanometer,r_boxes[1] * nanometer, r_boxes[2] * nanometer)
    first = 0
else:
    # Set initial positions.
    simulation.context.setPositions(pdb.positions)
    # Minimize the energy.
    if args.minimize:
        print "Minimization start, the energy is:", simulation.context.getState(getEnergy=True).getPotentialEnergy()
        simulation.minimizeEnergy()
        print "Minimization done, the energy is", simulation.context.getState(getEnergy=True).getPotentialEnergy()
        positions = simulation.context.getState(getPositions=True).getPositions()
        print "Minimized geometry is written to 'minimized.pdb'"
        PDBFile.writeModel(pdb.topology, positions, open('minimized.pdb','w'))
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

if args.eda_report_interval > 0:
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
        Reporter.report(simulation,simulation.context.getState(getPositions=True,getVelocities=True,getForces=True,getEnergy=True))

# This command actually does all of the computation.
simulation.step(args.production)

#=============================================#
#| Run analysis and save restart information |#
#=============================================#
simulation.reporters[0].analyze(simulation)
final_state = simulation.context.getState(getEnergy=True,getPositions=True,getVelocities=True,getForces=True)
Xfin = final_state.getPositions() / nanometer
Vfin = final_state.getVelocities() / nanometer * picosecond
Bfin = final_state.getPeriodicBoxVectors() / nanometer
bak(args.restart_filename)
logger.info("Restart information will be written to %s" % args.restart_filename)
with open(os.path.join(args.restart_filename),'w') as f: pickle.dump((Xfin, Vfin, Bfin),f)
print "Wall time: % .4f seconds" % (time.time() - t0)
print "Simulation speed: % .4f ns/day" % (86400*(first+args.production)*args.timestep*femtosecond/nanosecond/(time.time()-t0))
