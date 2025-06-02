import tkinter as tk
from tkinter import font

class GPUInputApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OracleGPU - Price Predictor")
        self.root.configure(bg="gray12")
        self.root.geometry("1920x1080")

        # Custom font setup
        self.title_font = font.Font(family="Helvetica", size=48, weight="bold")
        self.label_font = font.Font(family="Helvetica", size=24)
        self.entry_font = font.Font(family="Helvetica", size=16)

        # OracleGPU title
        tk.Label(
            root,
            text="OracleGPU",
            fg="cyan",
            bg="gray12",
            font=self.title_font
        ).pack(pady=(40, 60))

        # GPU name input
        tk.Label(
            root,
            text="Graphics Card Name:",
            fg="white",
            bg="gray12",
            font=self.label_font
        ).pack(pady=(0, 10))

        self.gpu_entry = tk.Entry(root, width=50, font=self.entry_font)
        self.gpu_entry.pack(pady=(0, 40))

        # Year input
        tk.Label(
            root,
            text="Year to Predict:",
            fg="white",
            bg="gray12",
            font=self.label_font
        ).pack(pady=(0, 10))

        self.year_entry = tk.Entry(root, width=50, font=self.entry_font)
        self.year_entry.pack(pady=(0, 40))

        # Submit Button
        tk.Button(
            root,
            text="Predict Price",
            command=self.submit,
            font=self.label_font,
            bg="black",
            fg="white",
            activebackground="gray25",
            activeforeground="cyan",
            padx=20,
            pady=10
        ).pack(pady=(10, 40))

        # Result Label
        self.result_label = tk.Label(
            root,
            text="",
            fg="lime",
            bg="gray12",
            font=self.label_font
        )
        self.result_label.pack(pady=(20, 10))

    def submit(self):
        self.gpu = self.gpu_entry.get().strip()
        self.year = self.year_entry.get().strip()

        if not self.gpu or not self.year.isdigit():
            self.result_label.config(text="⚠️ Please enter a valid GPU name and numeric year.")
        else:
            self.result_label.config(text=f"⏳ Predicting price for {self.gpu} in {self.year}...")
            print(f"User selected GPU: {self.gpu}, Year: {self.year}")
            self.prediction()

    def prediction(self):
        try:
            from training import training_process  # Ensure your training module is correctly placed
            pred_data = training_process(self.gpu, self.year)
            self.result_label.config(
                text=f"📈 Used: ${pred_data.usedPrice:.2f} | Retail: ${pred_data.retailPrice:.2f}"
            )
        except Exception as e:
            print(f"Prediction error: {e}")
            self.result_label.config(text="❌ Prediction failed. Please try again later.")

if __name__ == "__main__":
    root = tk.Tk()
    app = GPUInputApp(root)
    root.mainloop()

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
#plt.show() #Pull this window and attach it to tkinter window """