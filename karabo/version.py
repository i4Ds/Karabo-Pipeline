import os
###
__version__ = '0.11.1'
###


if os.getenv('BUILD_NIGHTLY', 'False') == 'True':
    __version__ = __version__ + '.nightly'

if __name__ == '__main__':
    import karabo
    print(karabo.__version__)
    
    