import tkinter as tk
from tkinter import ttk, messagebox
import requests
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from joblib import load
import warnings

warnings.filterwarnings("ignore")

class SolarANN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


API_KEY = "c7a1e5347b9b47dcdcefb11e9ab63ad9"

class SolarPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Solar Power Predictor")
        self.root.geometry("500x650")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.create_widgets()
        
        self.root.bind('<Return>', lambda _ : self.predict_power())
    
    def create_widgets(self):
        input_frame = ttk.Frame(self.root, padding="10")
        input_frame.grid(row=0, column=0, sticky="ew")
        
        ttk.Label(input_frame, text="Enter City:").pack(side=tk.LEFT, padx=5)
        self.city_entry = ttk.Entry(input_frame, width=25)
        self.city_entry.pack(side=tk.LEFT, padx=5)
        self.city_entry.focus()
        
        ttk.Button(input_frame, text="Predict", command=self.predict_power).pack(side=tk.LEFT, padx=5)
        
        self.result_frame = ttk.Frame(self.root, padding="10")
        self.result_frame.grid(row=1, column=0, sticky="nsew")
        
        self.alert_frame = ttk.Frame(self.root, padding="10")
        self.alert_frame.grid(row=2, column=0, sticky="ew")
        ttk.Label(self.alert_frame, text="Alerts & Recommendations:", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        self.alert_text = tk.Text(self.alert_frame, height=6, wrap=tk.WORD, bg="#FFF9E6")
        self.alert_text.pack(fill=tk.X)
        
        self.alert_text.tag_config("warning", foreground="red")
        self.alert_text.tag_config("recommend", foreground="blue")
    
    def predict_power(self):
        city = self.city_entry.get().strip()
        if not city:
            messagebox.showerror("Error", "Please enter a city name")
            return
        
        try:
            for widget in self.result_frame.winfo_children():
                widget.destroy()
            self.alert_text.delete(1.0, tk.END)
            
            geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&appid={API_KEY}"
            geo_data = requests.get(geo_url).json()
            
            if not geo_data:
                messagebox.showerror("Error", "City not found")
                return
            
            city_name_from_api = geo_data[0]['name'].lower() if geo_data else ""
            if city_name_from_api != city.lower():
                messagebox.showerror("Error", f"City '{city}' not found. Please check the name and try again.")
                return
            
            lat, lon = geo_data[0]['lat'], geo_data[0]['lon']
            
            weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            weather_data = requests.get(weather_url).json()
            
            temp = weather_data['main']['temp']
            clouds = weather_data['clouds']['all']
            current_hour = datetime.now().hour
            
            sunrise = datetime.fromtimestamp(weather_data['sys']['sunrise'])
            sunset = datetime.fromtimestamp(weather_data['sys']['sunset'])
            now = datetime.now()
            
            is_daylight = sunrise <= now <= sunset
            
            if is_daylight:
                day_duration = (sunset - sunrise).total_seconds()
                elapsed = (now - sunrise).total_seconds()
                irradiation = 1000 * np.sin(np.pi * elapsed/day_duration)
                irradiation *= (100 - min(clouds, 90))/100
            else:
                irradiation = 0
                
            X_input = np.array([[min(irradiation / 1000, 1.0), temp, temp + 5, current_hour]])
            X_scaled = x_scaler.transform(X_input)
            
            with torch.no_grad():
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                prediction = model(X_tensor).cpu().numpy()
                prediction_kw = y_scaler.inverse_transform(prediction)[0][0]
            
            final_pred = max(0, prediction_kw) if irradiation > 0.01 else 0
            
            self.show_results(city, lat, lon, now, sunrise, sunset, 
                            is_daylight, temp, clouds, irradiation, final_pred)
            
            self.generate_alerts(final_pred, irradiation, clouds, is_daylight)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def show_results(self, city, lat, lon, now, sunrise, sunset, is_daylight, temp, clouds, irradiation, final_pred):
        ttk.Label(self.result_frame, 
                 text=f"📍 Location: {city} ({lat:.2f}°N, {lon:.2f}°E)",
                 font=('Helvetica', 10, 'bold')).pack(anchor="w")
        
        ttk.Label(self.result_frame, 
                 text=f"🕒 Local Time: {now.strftime('%Y-%m-%d %H:%M')}").pack(anchor="w")
        
        ttk.Label(self.result_frame, 
                 text=f"Sunrise: {sunrise.strftime('%H:%M')} | Sunset: {sunset.strftime('%H:%M')}").pack(anchor="w")
        
        ttk.Label(self.result_frame, 
                 text=f"Daylight: {'Yes ' if is_daylight else 'No '}").pack(anchor="w")
        
        ttk.Label(self.result_frame, 
                 text=f"Temperature: {temp} °C").pack(anchor="w")
        
        ttk.Label(self.result_frame, 
                 text=f"Cloud Cover: {clouds}%").pack(anchor="w")
        
        ttk.Label(self.result_frame, 
                 text=f"Solar Irradiation: {irradiation:.1f} W/m²").pack(anchor="w")
        
        ttk.Label(self.result_frame, 
                 text=f"⚡ Predicted AC Power for Next Hour: {final_pred:.2f} W",
                 font=('Helvetica', 10, 'bold')).pack(anchor="w", pady=(10,0))
    
    def generate_alerts(self, final_pred, irradiation, clouds, is_daylight):
        if not is_daylight:
            self.alert_text.insert(tk.END, "🌙 Nighttime - No solar production expected\n", "warning")
            self.alert_text.insert(tk.END, "• Switch to grid/battery power\n", "recommend")
        elif irradiation < 100:
            self.alert_text.insert(tk.END, "⚠️ Low sunlight expected\n", "warning")
            self.alert_text.insert(tk.END, f"• Predicted output only {final_pred:.0f}W\n", "recommend")
            self.alert_text.insert(tk.END, "• Defer energy-intensive tasks\n", "recommend")
        elif final_pred > 800:
            self.alert_text.insert(tk.END, "☀️ High production expected!\n")
            self.alert_text.insert(tk.END, "• Good time to charge batteries\n", "recommend")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SolarANN(4).to(device)
model.load_state_dict(torch.load("./Solar_ANN/solar_ann_model.pth"))
x_scaler = load('./Solar_ANN/x_scaler.joblib')
y_scaler = load('./Solar_ANN/y_scaler.joblib')

if __name__ == "__main__":
    root = tk.Tk()
    app = SolarPredictorApp(root)
    root.mainloop()