import pip


def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])


import_or_install("numpy")
import_or_install("sklearn")
import_or_install("math")
import_or_install("itertools")
import_or_install("pandas")
import_or_install("os")
import_or_install("sys")
import_or_install("scipy")
import_or_install("sys")
import_or_install("matplotlib")
import_or_install("munkres")
import_or_install("seaborn")
import_or_install("itertools")
import_or_install("tkinter")
import_or_install("munkres")
import_or_install("sklearn")
