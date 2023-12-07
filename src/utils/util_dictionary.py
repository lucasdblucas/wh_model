
def get_value_key(dictionary, key_list):
    
    key = key_list.pop(0)
    if isinstance(dictionary[key], dict):
        return get_value_key(
            dictionary=dictionary[key],
            key_list=key_list
            )
    else: return dictionary[key]
    