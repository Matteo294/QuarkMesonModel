import os
import toml
import sys
import fileinput
import subprocess

cluster = sys.argv[1]


configurations = []

'''quark_masses = [0.0 - 0.15 * n for n in range(21)]
s = 1.0
for mq in quark_masses:
    configurations.append({ "physics": {"useMass": "true", "mass": 10.0, "g": 0.4, "kappa": 0.18, "lambda": 0.02, "cutFraction": s}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 10000.0, "ExportTime": 1.0, "burnCount": 200, "MeasureDriftCount": 60}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1432}, \
                        "fermions": {"yukawa_coupling": 0.6, "fermion_mass": mq, "driftMode": 0}, \
						"lattice": {"Nt": int(16/s), "Nx": int(16/s)} })'''

cutoffs = [1.0, 0.5, 0.25]
for s in cutoffs:
    configurations.append({ "physics": {"useMass": "true", "mass": -1.0*s*s, "g": 0.4*s*s, "kappa": 0.18, "lambda": 0.02, "cutFraction": s}, \
                        "langevin": {"averageEpsilon": 0.01, "MaxLangevinTime": 5000.0, "ExportTime": 1.0, "burnCount": 200, "MeasureDriftCount": 60}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1432}, \
                        "fermions": {"yukawa_coupling": 1.0*s, "fermion_mass": 1.0*s, "driftMode": 1, "WilsonParam": 0.0}, \
						"lattice": {"Nt": int(32/s), "Nx": int(32/s)} })

'''masses = [-1.0 + 0.2 * n for n in range(11)]
masses = [-1.0, -0.8]

s = 0.5
for m2 in masses:
    configurations.append({ "physics": {"useMass": "true", "mass": m2*s*s, "g": 0.4*s*s, "kappa": 0.18, "lambda": 0.02, "cutFraction": s}, \
                        "langevin": {"averageEpsilon": 0.01, "MaxLangevinTime": 5000.0, "ExportTime": 1.0, "burnCount": 200, "MeasureDriftCount": 60}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1432}, \
                        "fermions": {"yukawa_coupling": 0.5*s, "fermion_mass": 0.3*s, "driftMode": 1}, \
						"lattice": {"Nt": int(16/s), "Nx": int(16/s)} })'''

n_old_confs = max([int(d.replace("conf", "")) for d in os.listdir("./") if "conf" in d], default=0)
print("tot old configurations:", n_old_confs)



for count, conf in enumerate(configurations):

    count += n_old_confs
    
    
    print()
    print("=======================================================")
    print("Configuration", count + 1)
    
    dirname = "conf" + str(count + 1)
    
    # Create folde for this configuration
    process = subprocess.Popen("mkdir " + dirname, shell=True, stdout=subprocess.PIPE)
    process.wait()
    
    # Copy files into new folder
    process = subprocess.Popen("cp -r Yukawa_theory/*" + " " + dirname + "/", shell=True, stdout=subprocess.PIPE)
    process.wait()
    
     # Enter directory for this configuration
    os.chdir(dirname)
    
    # Load and modify toml params
    data = toml.load("./input.toml") 
    for section in conf.keys():
        if section in data.keys():
            print()
            for param in conf[section].keys():
                if param in data[section].keys():
                    print(section, param, conf[section][param])
                    data[section][param] = conf[section][param]
                    f = open("./input.toml",'w')
                    toml.dump(data, f)
                    f.close()
        if param == "Nt" or param == "Nx":
             print(section, param, conf[section][param])
    
     # Edit params.h file
    if conf["lattice"]["Nt"] != 0 and conf["lattice"]["Nx"] != 0:
        for line in fileinput.input("src/params.h", inplace=1):
            if "dimArray constexpr Sizes =" in line:
                line = line.replace("16, 16", str(conf["lattice"]["Nt"]) + ", " + str(conf["lattice"]["Nx"]))
            sys.stdout.write(line)
        process = subprocess.Popen("make clean && make -j", shell=True, stdout=subprocess.PIPE)
        process.wait()
    
    if cluster == "itp":
        filename = "runitp.sh"
        for line in fileinput.input(filename, inplace=1):
            if "cd" in line:
                line = line.replace("cd QuarkMesonModel/Yukawa_theory", "cd QuarkMesonModel/" + dirname)
            sys.stdout.write(line)
        process = subprocess.Popen("qsub runitp.sh", shell=True, stdout=subprocess.PIPE)
        process.wait()
    elif cluster == "bw":
        pass
        process = subprocess.Popen("sbatch runbw.sh", shell=True, stdout=subprocess.PIPE)
        process.wait()
        
    os.chdir("../")
    print()
    print("Done!")
    print("=======================================================")
    print()
    print()
