import os
import toml
import sys
import fileinput
import subprocess

cluster = sys.argv[1]

configurations = []

s = 1.0

configurations.append({ "physics": {"useMass": "true", "mass": 1.0, "g": 1.0, "kappa": 0.00, "lambda": 0.00, "cutFraction": 1.0}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 10000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 0}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.5, "fermion_mass": 0.15} })

configurations.append({ "physics": {"useMass": "true", "mass": 1.0, "g": 1.0, "kappa": 0.00, "lambda": 0.00, "cutFraction": 0.9}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 10000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 0}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.5, "fermion_mass": 0.15} })

configurations.append({ "physics": {"useMass": "true", "mass": 1.0, "g": 1.0, "kappa": 0.00, "lambda": 0.00, "cutFraction": 0.8}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 10000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 0}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.5, "fermion_mass": 0.15} })

configurations.append({ "physics": {"useMass": "true", "mass": 1.0, "g": 1.0, "kappa": 0.00, "lambda": 0.00, "cutFraction": 0.6}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 10000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 0}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.5, "fermion_mass": 0.15} })

configurations.append({ "physics": {"useMass": "true", "mass": 1.0, "g": 1.0, "kappa": 0.00, "lambda": 0.00, "cutFraction": 0.4}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 10000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 0}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.5, "fermion_mass": 0.15} })

configurations.append({ "physics": {"useMass": "true", "mass": 1.0, "g": 1.0, "kappa": 0.00, "lambda": 0.00, "cutFraction": 0.3}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 10000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 0}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.5, "fermion_mass": 0.15} })

configurations.append({ "physics": {"useMass": "true", "mass": 1.0, "g": 1.0, "kappa": 0.00, "lambda": 0.00, "cutFraction": 0.2}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 10000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 0}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.5, "fermion_mass": 0.15} })

configurations.append({ "physics": {"useMass": "true", "mass": 1.0, "g": 1.0, "kappa": 0.00, "lambda": 0.00, "cutFraction": 0.1}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 10000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 0}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.5, "fermion_mass": 0.15} })

configurations.append({ "physics": {"useMass": "true", "mass": 1.0, "g": 1.0, "kappa": 0.00, "lambda": 0.00, "cutFraction": 0.0}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 10000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 0}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.5, "fermion_mass": 0.15} })

                        






n_old_confs = max([int(d.replace("conf", "")) for d in os.listdir("./") if "conf" in d], default=0)
print("tot old configurations:", n_old_confs)



for count, conf in enumerate(configurations):

    count += n_old_confs
    
    dirname = "conf" + str(count + 1)
    
    # Create folde for this configuration
    process = subprocess.Popen("mkdir " + dirname, shell=True, stdout=subprocess.PIPE)
    process.wait()
    
    # Copy files into new folder
    process = subprocess.Popen("cp -r code/*" + " " + dirname + "/", shell=True, stdout=subprocess.PIPE)
    process.wait()
    
     # Enter directory for this configuration
    os.chdir(dirname)
    
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
        process = subprocess.Popen("qsub runitp.sh", shell=True, stdout=subprocess.PIPE)
        process.wait()
    elif cluster == "bw":
        process = subprocess.Popen("sbatch runbw.sh", shell=True, stdout=subprocess.PIPE)
        process.wait()
        
    os.chdir("../")
