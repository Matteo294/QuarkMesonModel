import os
import toml
import sys
import fileinput
import subprocess

cluster = sys.argv[1]

configurations = []

s = 0.5

configurations.append({ "physics": {"useMass": "true", "mass": 1.0*s*s, "g": 1.0*s*s, "kappa": 0.00, "lambda": 0.00, "cutFraction": s}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 1000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 10}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.4*s, "fermion_mass": 100*s} })

configurations.append({ "physics": {"useMass": "true", "mass": 1.0*s*s, "g": 1.0*s*s, "kappa": 0.00, "lambda": 0.00, "cutFraction": s}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 1000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 10}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.4*s, "fermion_mass": 50*s} })

configurations.append({ "physics": {"useMass": "true", "mass": 1.0*s*s, "g": 1.0*s*s, "kappa": 0.00, "lambda": 0.00, "cutFraction": s}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 1000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 10}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.4*s, "fermion_mass": 20*s} })

configurations.append({ "physics": {"useMass": "true", "mass": 1.0*s*s, "g": 1.0*s*s, "kappa": 0.00, "lambda": 0.00, "cutFraction": s}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 1000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 10}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.4*s, "fermion_mass": 10*s} })

configurations.append({ "physics": {"useMass": "true", "mass": 1.0*s*s, "g": 1.0*s*s, "kappa": 0.00, "lambda": 0.00, "cutFraction": s}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 1000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 10}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.4*s, "fermion_mass": 5*s} })

configurations.append({ "physics": {"useMass": "true", "mass": 1.0*s*s, "g": 1.0*s*s, "kappa": 0.00, "lambda": 0.00, "cutFraction": s}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 1000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 10}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.4*s, "fermion_mass": 1*s} })

configurations.append({ "physics": {"useMass": "true", "mass": 1.0*s*s, "g": 1.0*s*s, "kappa": 0.00, "lambda": 0.00, "cutFraction": s}, \
                        "langevin": {"averageEpsilon": 0.02, "MaxLangevinTime": 1000.0, "ExportTime": 1.0, "burnCount": 20, "MeasureDriftCount": 10}, \
                        "io": {"configFileName": "test.hdf", "export": "false", "timeSliceFileName": "slice.dat"}, \
                        "random": {"seed": 1234}, \
                        "fermions": {"yukawa_coupling": 0.4*s, "fermion_mass": 0.5*s} })


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
