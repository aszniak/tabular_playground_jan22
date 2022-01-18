import numpy as np
import matplotlib.pyplot as plt
from datetime import date

somedate = date.fromisoformat('2015-04-05')
timetuple = somedate.timetuple()
print(timetuple.tm_yday)


