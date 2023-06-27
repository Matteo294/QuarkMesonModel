from os import system
from os import chdir
import toml
import sys
import fileinput
import subprocess

cluster = sys.argv[1]

configurations = []

'''configurations.append({ "physics": {"useMass": "true", "mass": -2.0, "g": 2.0, "kappa": 0.00, "lambda": 0.00, "cutFraction": 1.0}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 1000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 10}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 2.0, "fermion_mass": 0.4} })'''

configurations.append({ "physics": {"useMass": "true", "mass": -2.0, "g": 2.0, "kappa": 0.00, "lambda": 0.00, "cutFraction": 1.0}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 1000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 10}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 2.0, "fermion_mass": 0.6} })

configurations.append({ "physics": {"useMass": "true", "mass": -2.0, "g": 2.0, "kappa": 0.00, "lambda": 0.00, "cutFraction": 1.0}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 1000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 10}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 2.0, "fermion_mass": 0.8} })

configurations.append({ "physics": {"useMass": "true", "mass": -2.0, "g": 2.0, "kappa": 0.00, "lambda": 0.00, "cutFraction": 1.0}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 1000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 10}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 2.0, "fermion_mass": 1.0} })

configurations.append({ "physics": {"useMass": "true", "mass": -2.0, "g": 2.0, "kappa": 0.00, "lambda": 0.00, "cutFraction": 1.0}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 1000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 10}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 2.0, "fermion_mass": 2.0} })

process = subprocess.Popen("rm -rf conf*", shell=True, stdout=subprocess.PIPE)
process.wait()

for count, conf in enumerate(configurations):
    
    dirname = "conf" + str(count + 1)
    
    # Create folde for this configuration
    process = subprocess.Popen("mkdir " + dirname, shell=True, stdout=subprocess.PIPE)
    process.wait()
    
    # Copy files into new folder
    process = subprocess.Popen("cp -r code/*" + " " + dirname + "/", shell=True, stdout=subprocess.PIPE)
    process.wait()
    
     # Enter directory for this configuration
    chdir(dirname)
    system("ls")
    
    # Load and modify toml params
    data = toml.load("./input.toml") 
    for section in conf.keys():
        if section in data.keys():
            for param in conf[section].keys():
                if param in data[section].keys():
                    data[section][param] = conf[section][param]
                    f = open("./input.toml",'w')
                    toml.dump(data, f)
                    f.close()
                else:
                    print(section, param, "--> parameter not found")
        else:
            print(section, param, "--> section not found")
    
    if cluster == "itp":
        filename = "runitp.sh"
        for line in fileinput.input(filename, inplace=1):
            if "cd" in line:
                line = line.replace("cd QuarkMesonModel/code", "cd QuarkMesonModel/" + dirname)
            sys.stdout.write(line)
        system("qsub runitp.sh")
    elif cluster == "bw":
        system("sbatch runbw.sh")
        
    chdir("../")

'''for s in cutoffs:
    dirname = "s_" + str(s).replace(".", "_")

    # Create folde for this configuration
    system("mkdir " + dirname)

    # Copy files into new folder
    system("cp -r ./mistake/* " + dirname + "/")

    # Enter directory for this configuration
    chdir(dirname)

    # Load and modify toml params
    data = toml.load("./input.toml") 
    data['physics']['cutFraction'] = float(s)

    # Write new data to toml
    f = open("./input.toml",'w')
    toml.dump(data, f)
    f.close()

    # Submit job
    
    if cluster == "itp":
        filename = "runitp.sh"
        for line in fileinput.input(filename, inplace=1):
            if "cd" in line:
                line = line.replace("cd QuarkMesonModel/mistake", "cd QuarkMesonModel/" + dirname)
            sys.stdout.write(line)
        system("qsub runitp.sh")
    elif cluster == "bw":
        system("sbatch runbw.sh")

    # Return to parent directory
    chdir("../")'''
