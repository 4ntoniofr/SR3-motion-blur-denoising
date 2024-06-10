import subprocess
import os

def entrenar_SR3(config_file):
    command = "python " + os.path.join("SR3", "sr.py") + " --config " + os.path.join("config", config_file) + " -p train"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.stdout.read().decode("utf-8"))

def main():
    config_file = input("Ingrese el nombre del archivo de configuración: ")
    entrenar_SR3(config_file)

if __name__ == "__main__":
    main()