import pickle

def save(object_, file_name):
    file = open(file_name, 'wb')
    pickle.dump(object_, file)
    file.close()


def load(file_name):    
    file = open(file_name, 'rb')
    object_ = pickle.load(file)
    file.close()
    return object_
