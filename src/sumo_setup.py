import os
import sys


def configure_sumo():
    if 'SUMO_HOME' not in os.environ:
        try:
            import sumo
        except ImportError:
            pass
        else:
            os.environ['SUMO_HOME'] = os.path.dirname(sumo.__file__)

    if 'SUMO_HOME' not in os.environ:
        sys.exit(
            "please declare environment variable 'SUMO_HOME' or install "
            "the eclipse-sumo package in the active Python environment"
        )

    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if os.path.isdir(tools) and tools not in sys.path:
        sys.path.append(tools)

    bin_dir = os.path.join(os.environ['SUMO_HOME'], 'bin')
    if os.path.isdir(bin_dir):
        path_entries = os.environ.get('PATH', '').split(os.pathsep)
        if bin_dir not in path_entries:
            os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')
