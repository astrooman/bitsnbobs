import matplotlib.pyplot as pplot
import numpy as np

top = 339.316973
bw = 33.333248 / 1024.0

dm = np.linspace(0.0, 2000.0, 400)
smearing = 8.3e+06 * dm * bw / ((top - bw / 2.0)**(3.0))
sampt = 128e-03

width1 = np.sqrt(1**2 + smearing**2 + sampt**2)
width2 = np.sqrt(2**2 + smearing**2 + sampt**2)
width3 = np.sqrt(5**2 + smearing**2 + sampt**2)

pplot.plot(dm, smearing, 'y--')
pplot.plot(dm, width1, label="1ms")
pplot.plot(dm, width2, label="2ms")
pplot.plot(dm, width3, label="5ms")
pplot.xlabel('DM [pc cm^-3]')
pplot.ylabel('Smearing [ms]')
pplot.minorticks_on()
pplot.grid(b=None, which='major')
pplot.grid(b=None, which='minor')
pplot.legend(loc=0)
pplot.show()
