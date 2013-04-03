"""
tinker_reporter.py: Runs TINKER alongside OpenMM dynamics for validation

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2012 Stanford University and the Authors.
Authors: Lee-Ping Wang
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
__author__ = "Lee-Ping Wang"
__version__ = "1.0"

import os, sys, shutil
from simtk.unit import *
from collections import OrderedDict, defaultdict
import itertools
import warnings
# Suppress warnings from PDB reading.
warnings.simplefilter("ignore")
from molecule import Molecule, isint, isfloat # Requires file format conversion class!
import subprocess
from subprocess import PIPE
import numpy as np
import re

def pvec(vec):
    """ Print a three-element vector in a Tinker .dyn format"""
    return ("% 26.16e% 26.16e% 26.16e" % (vec[0],vec[1],vec[2])).replace('e','D')

def EnergyDecomposition(Sim, print_thresh=1e-6, verbose=True):
    # The number of forces (interaction types)
    nfrc = Sim.system.getNumForces()
    # Loop through the interaction types and set the force group for each one.
    EnergyTerms = OrderedDict()
    TotalEnergy = Sim.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
    EnergyTerms['Potential Energy'] = TotalEnergy
    if verbose:
        print "%-35s % 16.8f" % ('Potential Energy', TotalEnergy)
    # for i in range(nfrc):
    #     Sim.system.getForce(i).setForceGroup(i)
    for i in range(nfrc):
        print "Doing energy decomposition for Force named", Sim.system.getForce(i).__class__.__name__
        Energy = Sim.context.getState(getEnergy=True,groups=2**i).getPotentialEnergy() / kilojoules_per_mole
        if abs(Energy) > print_thresh:
            EnergyTerms[Sim.system.getForce(i).__class__.__name__] = Energy
            if verbose:
                print "%-35s" % Sim.system.getForce(i).__class__.__name__,
                print "% 16.8f" % Energy
    # for i in range(nfrc):
    #     Sim.system.getForce(i).setForceGroup(0)
    return EnergyTerms

def ForceDecomposition(Sim, print_thresh=1e-6, verbose=False):
    # The number of forces (interaction types)
    nfrc = Sim.system.getNumForces()
    # Loop through the interaction types and set the force group for each one.
    ForceTerms = OrderedDict()
    TotalForce = Sim.context.getState(getForces=True).getForces() / kilojoules_per_mole * nanometer
    TotalForce = np.array(list(itertools.chain(*[list(i) for i in TotalForce])))
    ForceTerms['Total Force'] = TotalForce
    # for i in range(nfrc):
    #     Sim.system.getForce(i).setForceGroup(i)
    for i in range(nfrc):
        print "Doing force decomposition for Force named", Sim.system.getForce(i).__class__.__name__
        Force = Sim.context.getState(getForces=True,groups=2**i).getForces() / kilojoules_per_mole * nanometer
        Force = np.array(list(itertools.chain(*[list(j) for j in Force])))
        if max(abs(Force)) > print_thresh:
            ForceTerms[Sim.system.getForce(i).__class__.__name__] = Force
    # for i in range(nfrc):
    #     Sim.system.getForce(i).setForceGroup(0)
    return ForceTerms

def _exec(command, print_to_screen = False, outfnm = None, logfnm = None, stdin = None, print_command = True):
    """Runs command line using subprocess, optionally returning stdout.
    Options:
    command (required) = Name of the command you want to execute
    outfnm (optional) = Name of the output file name (overwritten if exists)
    logfnm (optional) = Name of the log file name (appended if exists), exclusive with outfnm
    stdin (optional) = A string to be passed to stdin, as if it were typed (use newline character to mimic Enter key)
    print_command = Whether to print the command.
    """
    append_to_file = (logfnm != None)
    write_to_file = (outfnm != None)
    if append_to_file:
        f = open(logfnm,'a')
    elif write_to_file:
        f = open(outfnm,'w')
    if print_command:
        print "Executing process: \x1b[92m%-50s\x1b[0m%s%s" % (' '.join(command) if type(command) is list else command,
                                                               " Output: %s" % logfnm if logfnm != None else "",
                                                               " Stdin: %s" % stdin.replace('\n','\\n') if stdin != None else "")
        if append_to_file or write_to_file:
            print >> f, "Executing process: %s%s" % (command, " Stdin: %s" % stdin.replace('\n','\\n') if stdin != None else "")
    if stdin == None:
        p = subprocess.Popen(command, shell=(type(command) is str), stdout = PIPE, stderr = PIPE)
        if print_to_screen:
            Output = []
            Error = []
            while True:
                line = p.stdout.readline()
                try:
                    Error.append(p.stderr.readline())
                except: pass
                if not line:
                    break
                print line,
            Output.append(line)
            print Error
        else:
            Output, Error = p.communicate()
    else:
        p = subprocess.Popen(command, shell=(type(command) is str), stdin = PIPE, stdout = PIPE, stderr = PIPE)
        Output, Error = p.communicate(stdin)
    if logfnm != None or outfnm != None:
        f.write(Output)
        f.close()
    if p.returncode != 0:
        print "Received an error message:"
        print Error
        warn_press_key("%s gave a return code of %i (it may have crashed)" % (command, p.returncode))
    return Output

def printcool(text,sym="#",bold=False,color=2,bottom='-',minwidth=50):
    """Cool-looking printout."""
    text = text.split('\n')
    width = max(minwidth,max([len(line) for line in text]))
    bar = ''.join([sym for i in range(width + 8)])
    print '\n'+bar
    for line in text:
        padleft = ' ' * ((width - len(line)) / 2)
        padright = ' '* (width - len(line) - len(padleft))
        print "%s\x1b[%s9%im%s" % (''.join([sym for i in range(3)]), bold and "1;" or "", color, padleft),line,"%s\x1b[0m%s" % (padright, ''.join([sym for i in range(3)]))
    print bar
    return re.sub(sym,bottom,bar)

def printcool_dictionary(Dict,title="General options",bold=False,color=2,keywidth=25,topwidth=50):
    """Nice way to print out keys/values in a dictionary."""
    bar = printcool(title,bold=bold,color=color,minwidth=topwidth)
    def magic_string(str):
        # This cryptic command returns a string with the number of characters specified as a variable. :P
        return eval("\'%%-%is\' %% '%s'" % (keywidth,str.replace("'","\\'").replace('"','\\"')))
    if isinstance(Dict, OrderedDict):
        print '\n'.join(["%s %s " % (magic_string(key),str(Dict[key])) for key in Dict if Dict[key] != None])
    else:
        print '\n'.join(["%s %s " % (magic_string(key),str(Dict[key])) for key in sorted([i for i in Dict]) if Dict[key] != None])
    print bar

class TinkerReporter(object):
    """TinkerReporter

    To use it, create a TinkerReporter, then add it to the Simulation's list of reporters.
    """

    def __init__(self, file, reportInterval, simulation, xyzfile, pbc=True, pmegrid=None, tinkerpath = ''):
        """Create a TinkerReporter.

        Parameters:
         - tinkerpath (string) The
         - file (string) The file to write to
         - reportInterval (int) The interval (in time steps) at which to write frames
        """
        print "Initializing Tinker Reporter"
        self._reportInterval = reportInterval
        self._openedFile = isinstance(file, str)
        if self._openedFile:
            self._out = open(file, 'w')
        else:
            self._out = file
        self._pbc = pbc
        self._pmegrid = pmegrid
        self._tinkerpath = tinkerpath
        self._simulation = simulation

        self.xyzfile = xyzfile
        self.pdbfile = os.path.splitext(xyzfile)[0] + '.pdb'
        self.keyfile = os.path.splitext(xyzfile)[0] + '.key'
        #self.prmfile = os.path.splitext(xyzfile)[0] + '.prm'
        self.dynfile = os.path.splitext(xyzfile)[0] + '.dyn'

        print "Loading Tinker xyz file"
        if not os.path.exists(self.xyzfile):
            raise IOError('You need a Tinker .xyz')
        if not os.path.exists(self.keyfile):
            raise IOError('You need a Tinker .key file with the same base name as the .xyz file')
        #if not os.path.exists(self.prmfile):
        #    raise IOError('You need a Tinker .prm file with the same base name as the .xyz file')
        if not os.path.exists(self.pdbfile):
            raise IOError('You need a .pdb file with the same base name as the .xyz file (the same one you used to start the simulation)')
        self.M = Molecule(self.xyzfile,ftype='tinker')
        self.comm = open(self.xyzfile).readlines()[0]
        print "Done"

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters:
         - simulation (Simulation) The Simulation to generate a report for
        Returns: A five element tuple.  The first element is the number of steps until the
        next report.  The remaining elements specify whether that report will require
        positions, velocities, forces, and energies respectively.
        """
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, True, True, True)

    def report(self, simulation, state):
        """Generate a report.

        Parameters:
         - simulation (Simulation) The Simulation to generate a report for
         - state (State) The current state of the simulation
        """

        RunDynamic = 0
        pos = state.getPositions() / angstrom
        if RunDynamic:
            # Write a dyn file.
            dynout = open(self.dynfile,'w')
            print >> dynout, ' Number of Atoms and Title :'
            print >> dynout, self.comm,
            print >> dynout, ' Periodic Box Dimensions :'
            print >> dynout, pvec([state.getPeriodicBoxVectors()[i][i] / angstrom for i in range(3)])
            print >> dynout, pvec([90,90,90])
            print >> dynout, ' Current Atomic Positions :'
            for i in range(len(pos)):
                print >> dynout, pvec(pos[i])
            print >> dynout, ' Current Atomic Velocities :'
            vel = state.getVelocities() / angstrom * picosecond
            for i in range(len(vel)):
                print >> dynout, pvec(vel[i])
            print >> dynout, ' Current Atomic Accelerations :'
            for i in range(len(vel)):
                print >> dynout, pvec([0,0,0])
            print >> dynout, ' Alternate Atomic Accelerations :'
            for i in range(len(vel)):
                print >> dynout, pvec([0,0,0])
            dynout.close()
            o = _exec('%s %s 1 1e-5 1 1' % (os.path.join(self._tinkerpath,'dynamic'), self.xyzfile), print_command = False)
            for line in o.split('\n'):
                if 'Total Energy' in line:
                    total = float(line.split()[2]) * 4.184
                elif 'Potential Energy' in line:
                    pot = float(line.split()[2]) * 4.184
                elif 'Kinetic Energy' in line:
                    kin = float(line.split()[2]) * 4.184
                elif 'Temperature' in line:
                    temp = float(line.split()[1])
                elif 'Pressure' in line:
                    pres = float(line.split()[1])
                elif 'Density' in line:
                    dens = float(line.split()[1]) * 1000
            print "(Tinker dynamic)     % 13.5f % 13.5f % 13.5f % 13.5f % 13.5f" % (temp, pot, kin, total, pres)

        RunAnalyze = 1
        RunGrad = 1
        if RunAnalyze or RunGrad:
            # Write a xyz file.
            self.M.xyzs = [pos]
            self.M.boxes = [[state.getPeriodicBoxVectors()[0][0] / angstrom,state.getPeriodicBoxVectors()[1][1] / angstrom,state.getPeriodicBoxVectors()[2][2] / angstrom]]
            self.M.write('temp.xyz',ftype='tinker')
            keyout = open('temp.key','w')
            for line in open(self.keyfile).readlines():
                if 'a-axis' in line.lower(): pass
                elif 'b-axis' in line.lower(): pass
                elif 'c-axis' in line.lower(): pass
                elif 'pme-grid' in line.lower(): pass
                else: print >> keyout, line,
            if self._pbc:
                print >> keyout, 'a-axis % .6f' % (state.getPeriodicBoxVectors()[0][0] / angstrom)
                print >> keyout, 'b-axis % .6f' % (state.getPeriodicBoxVectors()[1][1] / angstrom)
                print >> keyout, 'c-axis % .6f' % (state.getPeriodicBoxVectors()[2][2] / angstrom)
                if self._pmegrid != None:
                    print >> keyout, 'pme-grid %i %i %i' % (self._pmegrid[0], self._pmegrid[1], self._pmegrid[2])
            keyout.close()
            #shutil.copy(self.prmfile,'temp.prm')

        if RunAnalyze:
            # Actually I should use testgrad for this (maybe later.)
            print "Running TINKER analyze program"
            o = _exec('%s temp.xyz E' % os.path.join(self._tinkerpath,'analyze'), print_command = False)
            # This is a dictionary of TINKER energy terms to OpenMM energy terms
            # I'm making the assumption that OpenMM energy terms are less finely divided.
            T2O = OrderedDict([('Total Potential Energy', 'Potential Energy'),
                               ('Bond Stretching','AmoebaBondForce'),
                               ('Angle Bending','AmoebaAngleForce'),
                               ('Stretch-Bend','AmoebaStretchBendForce'),
                               ('Out-of-Plane Bend','AmoebaOutOfPlaneBendForce'),
                               ('Torsional Angle','PeriodicTorsionForce'),
                               ('Urey-Bradley','HarmonicBondForce'),
                               ('Van der Waals','AmoebaVdwForce'),
                               ('Atomic Multipoles','AmoebaMultipoleForce'),
                               ('Polarization','AmoebaMultipoleForce')])

            Tinker_Energy_Terms = defaultdict(float)

            for line in o.split('\n'):
                s = line.split()
                for TTerm, OTerm in T2O.items():
                    # Very strict parsing.
                    tts = TTerm.split()
                    if len(s) == len(tts) + 2 and s[:len(tts)] == tts:
                        Tinker_Energy_Terms[OTerm] += float(s[len(tts)]) * 4.184
                    elif TTerm == 'Total Potential Energy' and s[:len(tts)] == tts:
                        Tinker_Energy_Terms[OTerm] = float(s[4]) * 4.184

            OpenMM_Energy_Terms = EnergyDecomposition(self._simulation)

            PrintDict = OrderedDict()
            for key, val in OpenMM_Energy_Terms.items():
                o = val
                t = Tinker_Energy_Terms[key]
                PrintDict[key] = " % 13.5f - % 13.5f = % 13.5f" % (o, t, o-t)
            printcool_dictionary(PrintDict, title="Energy Terms (kJ/mol): %13s - %13s = %13s" % ("OpenMM", "Tinker", "Difference"))


        if RunGrad:
            TFrc = OrderedDict()
            TFrcO = OrderedDict()
            TFrcTypes = []
            Mode = 0
            TFrcNum = 0
            print "Running TINKER testgrad program"
            o = _exec('%s temp.xyz Y N Y' % os.path.join(self._tinkerpath,'testgrad'), print_command = False)

            # The following is copied over from TINKER source code.
            # c     desum   total energy Cartesian coordinate derivatives
            # c     deb     bond stretch Cartesian coordinate derivatives
            # c     dea     angle bend Cartesian coordinate derivatives
            # c     deba    stretch-bend Cartesian coordinate derivatives
            # c     deub    Urey-Bradley Cartesian coordinate derivatives
            # c     deaa    angle-angle Cartesian coordinate derivatives
            # c     deopb   out-of-plane bend Cartesian coordinate derivatives
            # c     deopd   out-of-plane distance Cartesian coordinate derivatives
            # c     deid    improper dihedral Cartesian coordinate derivatives
            # c     deit    improper torsion Cartesian coordinate derivatives
            # c     det     torsional Cartesian coordinate derivatives
            # c     dept    pi-orbital torsion Cartesian coordinate derivatives
            # c     debt    stretch-torsion Cartesian coordinate derivatives
            # c     dett    torsion-torsion Cartesian coordinate derivatives
            # c     dev     van der Waals Cartesian coordinate derivatives
            # c     dec     charge-charge Cartesian coordinate derivatives
            # c     decd    charge-dipole Cartesian coordinate derivatives
            # c     ded     dipole-dipole Cartesian coordinate derivatives
            # c     dem     multipole Cartesian coordinate derivatives
            # c     dep     polarization Cartesian coordinate derivatives
            # c     der     reaction field Cartesian coordinate derivatives
            # c     des     solvation Cartesian coordinate derivatives
            # c     delf    metal ligand field Cartesian coordinate derivatives
            # c     deg     geometric restraint Cartesian coordinate derivatives
            # c     dex     extra energy term Cartesian coordinate derivatives

            T2O = OrderedDict([('EB', 'AmoebaBondForce'),
                               ('EA', 'AmoebaAngleForce'),
                               ('EUB','HarmonicBondForce'),
                               ('EV', 'AmoebaVdwForce'),
                               ('EM', 'AmoebaMultipoleForce'),
                               ('EP', 'AmoebaMultipoleForce')])
            for line in o.split('\n'):
                if "Cartesian Gradient Breakdown by Individual Components" in line:
                    Mode = 1
                elif "Cartesian Gradient Breakdown over Individual Atoms" in line:
                    Mode = 0
                if Mode:
                    if 'd E' in line:
                        for wrd in re.findall("d E[A-Z]+",line):
                            TFrc[wrd.split()[1]] = []
                            TFrcTypes.append(wrd.split()[1])
                    else:
                        s = line.split()
                        for w in s:
                            if isfloat(w) and not isint(w):
                                TFrc[TFrcTypes[TFrcNum%len(TFrcTypes)]].append(float(w))
                                TFrcNum += 1
            for TFrcType in TFrc:
                TFrcArray = np.array(TFrc[TFrcType])
                MaxF = max(abs(TFrcArray))
                if MaxF != 0.0 and TFrcType not in T2O:
                    raise Exception('Oopsh! The Tinker force type %s needs to correspond to one of the AMOEBA forces.' % TFrcType)
                if 'Total Force' in TFrcO:
                    TFrcO['Total Force'] += -41.84*TFrcArray.copy()
                else:
                    TFrcO['Total Force'] = -41.84*TFrcArray.copy()
                if TFrcType in T2O:
                    if T2O[TFrcType] in TFrcO:
                        #print "Incrementing Tinker Force %s into OpenMM Force %s" % (TFrcType, T2O[TFrcType])
                        TFrcO[T2O[TFrcType]] += -41.84*TFrcArray.copy()
                    else:
                        #print "Copying Tinker Force %s into OpenMM Force %s" % (TFrcType, T2O[TFrcType])
                        TFrcO[T2O[TFrcType]] = -41.84*TFrcArray.copy()

            OFrc = ForceDecomposition(self._simulation)

            for key, val in OFrc.items():

                Fo = val.reshape(-1,3)
                Ft = TFrcO[key].reshape(-1,3)
                rmsf = np.sqrt(np.mean(TFrcO[key] ** 2))
                rmse = np.sqrt(np.mean((TFrcO[key]-val) ** 2))
                maxa = 0
                maxdf = 0
                ErrThresh = 1e-2
                BadAtoms = []
                ForceDev = []
                for a in range(len(Fo)):
                    fo = Fo[a]
                    ft = Ft[a]
                    df = fo-ft
                    ForceDev.append(df)
                    if np.linalg.norm(df) > maxdf:
                        maxdf = np.linalg.norm(df)
                        maxa = a
                    if np.linalg.norm(df) / rmsf > ErrThresh:
                        BadAtoms.append(a)
                ForceDev = np.array(ForceDev)
                ForceMax = np.max(np.max(np.abs(ForceDev)))
                ForceDev /= ForceMax
                printcool("Checking Force : %s" % key)
                print "RMS Force = % .6f" % rmsf
                print "RMS error = % .6f (%.4f%%)" % (rmse, 100*rmse/rmsf)
                print "Max error = % .6f (%.4f%%) on atom %i" % (maxdf, 100*maxdf/rmsf, maxa)
                if len(BadAtoms) > 0 and key != 'Total Force':
                    MBad = self.M[0]
                    print "Printing coordinates and force errors..."
                    MBad.write('%s.%04i.coord.xyz' % (key,self._simulation.currentStep))
                    MBad.xyzs = [ForceDev]
                    MBad.write('%s.%04i.dforce.xyz' % (key,self._simulation.currentStep))

 # Total Potential Energy :         -11774.66948293 Kcal/mole

 # Energy Component Breakdown :           Kcal/mole      Interactions

 # Bond Stretching                     700.95903376           2048
 # Angle Bending                       280.53183695           1024
 # Urey-Bradley                        -17.15154506           1024
 # Van der Waals                      6125.15094324         444788
 # Atomic Multipoles                -14366.63061078         216556
 # Polarization                      -4497.52914104         216556

    def __del__(self):
        if self._openedFile:
            self._out.close()
