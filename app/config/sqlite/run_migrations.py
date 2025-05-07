import os
import subprocess

os.environ["PICCOLO_CONF"] = "app.config.sqlite.piccolo_conf"

# env_module_name = os.environ.get(ENVIRONMENT_VARIABLE, None)
def run_command(command):
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output


print(run_command("piccolo migrations new sqlite --auto"))

