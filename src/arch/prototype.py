

class Storage():
    inputData:MutableData
    outputData:MutableData
    [data]:Data

    # by default Immutable
    class Data():
        def read()

    class MutableData(Data):
        def write()

    class Iterator():
        def getData()

class Architecture():
    class Node():
        [args]:Iterator
        res:Iterator #MutableDataIterator
        funct:Function

    [prev_arch]:Architecture
    node:Node

    def rebase(new_prev_arch, node_args)

class NeuralNet():
    a:Architecture
    d:Storage
