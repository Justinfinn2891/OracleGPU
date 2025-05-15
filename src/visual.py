
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

#Load the full data (dates + prices)
dates_str, _, price1, price2 = np.loadtxt(
    '../data/data.csv', 
    delimiter=',', 
    dtype=str, 
    skiprows=1, 
    usecols=(0,1,2,3), 
    unpack=True
)

#Convert date strings to datetime objects for plotting
dates = [datetime.strptime(date, '%m-%d-%y') for date in dates_str] # Convert date strings to datetime objects

#Convert prices from strings to float
price1 = price1.astype(float) # New price
price2 = price2.astype(float) # Used price

plt.figure(figsize=(12, 6))
plt.plot(dates, price1, label='New Price') # Plot new price
plt.plot(dates, price2, label='Used Price') # Plot used price


plt.xlabel('Date') # X-axis label
plt.ylabel('Price ($)') # Y-axis label
plt.title('GPU Prices Over Time') # Title of the plot
plt.legend() # Add legend
plt.grid(True) # Add grid for better readability

plt.xticks(rotation=45)  # Rotate dates for better readability
plt.tight_layout()       # Adjust layout so labels fit
plt.show()


#data = np.loadtxt('../data/data.csv', delimiter=',', skiprows=1, usecols=(2, 3))

# Xvalue for retail prices
# Yvalue for date



# XUvalue for used prices
# YUvalue for date 


#Example below 
# xpoints, ypoints (years, prices)

#exampleY = [2003, 2004, 2005, 2006]
##exampleX = [425, 231, 453, 646]
#xpoints = np.array([0,6])
#ypoints = np.array([0,250])

##plt.plot(exampleY, exampleX, 'o-g') # plot the points
#plt.show() #Pull this window and attach it to tkinter window 