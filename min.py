#!/usr/bin/env python

from forcebalance.molecule import *
from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *

# Center the positions before doing anything
M = Molecule(sys.argv[1], build_topology=False)
M.xyzs[0] -= np.mean(M.xyzs[0], axis=0)
M[0].write(".temp.pdb")

pdb = PDBFile(".temp.pdb")

ff = ForceField(*sys.argv[2:])
#ff = ForceField('amber-fb15.xml', 'tip3pfb.xml')

def make_cef(k):
    # Create a Custom Force
    # This custom force does not work because of PBC.
    # cef = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    # This is a new OpenMM feature. :)
    cef = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    cef.addPerParticleParameter("k")
    cef.addPerParticleParameter("x0")
    cef.addPerParticleParameter("y0")
    cef.addPerParticleParameter("z0")
    for i in range(M.na):
        if M.resname[i] != 'HOH':
            cef.addParticle(i, [k] + list(pdb.positions[i] / nanometer))
        else:
            cef.addParticle(i, [0.0] + list(pdb.positions[i] / nanometer))
    return cef

pos = pdb.positions
plat = Platform.getPlatformByName('CUDA')
#plat.setPropertyDefaultValue('CudaDeviceIndex','1')
for k in [100000, 10000, 1000]:
    system = ff.createSystem(pdb.topology, nonbondedMethod=PME, constraints=HBonds, rigidWater=True)
    cef = make_cef(k)
    system.addForce(cef)
    print "Created System with restraint", k
    integ = VerletIntegrator(0.00001*picosecond)
    sim = Simulation(pdb.topology, system, integ, plat)
    sim.context.setPositions(pos)
    print "Created Simulation, Potential = ", sim.context.getState(getEnergy=True).getPotentialEnergy()
    sim.minimizeEnergy()
    print "Energy Minimized, Potential = ", sim.context.getState(getEnergy=True).getPotentialEnergy()
    pos = sim.context.getState(getPositions=True).getPositions()

M.xyzs = [np.array(pos.value_in_unit(angstrom))]
M.write(os.path.splitext(os.path.split(sys.argv[1])[1])[0]+"_min.pdb")
# M.write(os.path.join("03.Minimized", os.path.split(sys.argv[1])[1]))
print "Wrote new PDB"
