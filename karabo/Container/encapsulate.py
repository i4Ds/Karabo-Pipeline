def createDockerEnvironment(conda_env_name, conda_env_file):
    if not conda_env_file:
        env = getCondaEnvs()
        print(env)


def getCondaEnvs():
    import json
    from subprocess import run
    proc = run(["conda", "env", "list", "--json"], text=True, capture_output=True)
    return json.load(proc.stdout)


if __name__ == '__main__':
    createDockerEnvironment("","")