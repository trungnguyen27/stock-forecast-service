import hashlib
import json

def get_prediction_model_hash(model_params):
    sha = hashlib.sha256()
    jsonobj = json.dumps(model_params, sort_keys=True, default=str)
    print('HASHING: %s' %jsonobj)
    sha.update(jsonobj.encode('utf-8'))
    hashed = sha.hexdigest()
    print('HASHED: %s' %hashed)
    return hashed