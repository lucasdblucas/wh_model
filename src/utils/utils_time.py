import datetime as dt


def convert_seconds(seconds_entry):
    # calculate hours, minutes, and remaining seconds
    hours = int(seconds_entry // 3600)
    minutes = int((seconds_entry % 3600) // 60)
    seconds = int(seconds_entry % 60)
    
    # return formatted string
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_time_string():
    
    now = dt.datetime.now()
    
    return "{}y-{}m-{}d-{}h".format(now.year, now.month, now.day, now.hour)