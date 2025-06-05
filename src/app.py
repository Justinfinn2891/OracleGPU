import tkinter as tk
from tkinter import font
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class GPUInputApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OracleGPU - Price Predictor")
        self.root.configure(bg="gray12")
        self.root.geometry("1280x800")

        # Fonts
        self.title_font = font.Font(family="Helvetica", size=48, weight="bold")
        self.label_font = font.Font(family="Helvetica", size=24)
        self.entry_font = font.Font(family="Helvetica", size=16)

        # Title
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

        # Predict Button
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

        # Result label
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
            from training import training_process
            predictor = training_process(self.gpu, self.year)

            self.result_label.config(
                text=f"📈 Used: ${predictor.usedPrice:.2f} | Retail: ${predictor.retailPrice:.2f}"
            )

            # Remove existing graphs if any
            for canvas_attr in ['canvas1', 'canvas2']:
                if hasattr(self, canvas_attr):
                    getattr(self, canvas_attr).get_tk_widget().destroy()

            # ---------------- Plot 1: Epoch Loss ----------------
            fig1, ax1 = plt.subplots(figsize=(6, 3))
            ax1.plot(predictor.loss_history, color='magenta')
            ax1.set_title('Training Loss Over Epochs')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True)

            self.canvas1 = FigureCanvasTkAgg(fig1, master=self.root)
            self.canvas1.draw()
            self.canvas1.get_tk_widget().pack(pady=10)

            # ---------------- Plot 2: Predicted Prices ----------------
            years = list(range(2020, int(self.year) + 1))
            retail_prices = [predictor.retailPrice] * len(years)
            used_prices = [predictor.usedPrice] * len(years)

            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.plot(years, retail_prices, label='Retail Price', color='cyan')
            ax2.plot(years, used_prices, label='Used Price', color='lime')
            ax2.set_title('Predicted GPU Prices')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Price ($)')
            ax2.legend()
            ax2.grid(True)

            self.canvas2 = FigureCanvasTkAgg(fig2, master=self.root)
            self.canvas2.draw()
            self.canvas2.get_tk_widget().pack(pady=10)

        except Exception as e:
            print(f"Prediction error: {e}")
            self.result_label.config(text="❌ Prediction failed. Please try again later.")

if __name__ == "__main__":
    root = tk.Tk()
    app = GPUInputApp(root)
    root.mainloop()