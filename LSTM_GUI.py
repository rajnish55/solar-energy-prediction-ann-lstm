import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import requests
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime


class SolarLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    

API_KEY = "c7a1e5347b9b47dcdcefb11e9ab63ad9"


class SolarPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Solar Power Predictor (Next 6 Hours)")
        self.root.geometry("1920x1080")
        
        # Load model and scalers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SolarLSTM(input_size=5).to(self.device)
        self.model.load_state_dict(torch.load('./Solar_LSTM/solar_lstm_model.pth', map_location=self.device))
        self.model.eval()
        self.x_scaler = joblib.load('./Solar_LSTM/x_scaler.joblib')
        self.y_scaler = joblib.load('./Solar_LSTM/y_scaler.joblib')
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="City:").pack(side=tk.LEFT, padx=5)
        self.city_entry = ttk.Entry(input_frame, width=25)
        self.city_entry.pack(side=tk.LEFT, padx=5)
        self.city_entry.insert(0, "")
        self.city_entry.focus()
        self.city_entry.bind('<Return>', lambda event: self.predict_power())
        
        ttk.Button(input_frame, text="Predict", command=self.predict_power).pack(side=tk.LEFT, padx=10)
        
        self.current_frame = ttk.LabelFrame(main_frame, text="Current Conditions", padding=10)
        self.current_frame.pack(fill=tk.X, pady=5)
        
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        plot_container = ttk.Frame(content_frame)
        plot_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.plot_frame = ttk.LabelFrame(plot_container, text="6-Hour Forecast", padding=10)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        alert_container = ttk.Frame(content_frame, width=300)
        alert_container.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        self.alert_frame = ttk.LabelFrame(alert_container, text="Alerts", padding=10)
        self.alert_frame.pack(fill=tk.X, pady=5)
        
        self.alert_text = tk.Text(self.alert_frame, height=10, wrap=tk.WORD, bg="#FFE6E6")
        self.alert_text.pack(fill=tk.X)
        self.alert_text.tag_config("alert", foreground="red")
        
        self.recommend_frame = ttk.LabelFrame(alert_container, text="Recommendations", padding=10)
        self.recommend_frame.pack(fill=tk.X, pady=5)
        
        self.recommend_text = tk.Text(self.recommend_frame, height=10, wrap=tk.WORD, bg="#E6FFE6")
        self.recommend_text.pack(fill=tk.X)
        self.recommend_text.tag_config("recommend", foreground="blue")
        
    def clear_display(self):
        for widget in self.current_frame.winfo_children():
            widget.destroy()
        
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.canvas.draw()

        self.alert_text.delete(1.0, tk.END)
        self.recommend_text.delete(1.0, tk.END)
        
    def predict_power(self):
        city = self.city_entry.get().strip()
        if not city:
            messagebox.showerror("Error", "Please enter a city name")
            return
        
        try:
            self.clear_display()
            
            API_KEY = "c7a1e5347b9b47dcdcefb11e9ab63ad9"
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
            
            forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            forecast_data = requests.get(forecast_url).json()
            
            times, hours = [], []
            irradiations, amb_temps, mod_temps = [], [], []
            is_daylight_list, weather_conditions = [], []
            
            now = datetime.now()
            sunrise = datetime.fromtimestamp(forecast_data['city']['sunrise'])
            sunset = datetime.fromtimestamp(forecast_data['city']['sunset'])
            
            for i in range(6):
                forecast = forecast_data['list'][i]
                time = datetime.fromtimestamp(forecast['dt'])
                is_daylight = sunrise <= time <= sunset
                
                if is_daylight:
                    day_duration = (sunset - sunrise).total_seconds()
                    elapsed = (time - sunrise).total_seconds()
                    irradiation = 1000 * np.sin(np.pi * elapsed / day_duration)
                    cloud_factor = (100 - min(forecast['clouds']['all'], 90)) / 100
                    irradiation *= cloud_factor
                else:
                    irradiation = 0
                
                amb_temp = forecast['main']['temp']
                mod_temp = amb_temp + 5
                weather_desc = forecast['weather'][0]['description']
                
                times.append(time)
                hours.append(time.hour)
                irradiations.append(irradiation)
                amb_temps.append(amb_temp)
                mod_temps.append(mod_temp)
                is_daylight_list.append(is_daylight)
                weather_conditions.append(weather_desc)
            
            hours_sin = np.sin(2 * np.pi * np.array(hours) / 24)
            hours_cos = np.cos(2 * np.pi * np.array(hours) / 24)
            X = np.column_stack([irradiations, amb_temps, mod_temps, hours_sin, hours_cos])
            X_scaled = self.x_scaler.transform(X).reshape(1, 6, 5)
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                pred_scaled = self.model(X_tensor).cpu().numpy()
                predictions = self.y_scaler.inverse_transform(pred_scaled)[0]
            
            predictions = [pred if is_daylight else 0 for pred, is_daylight in zip(predictions, is_daylight_list)]
            
            self.display_results(forecast_data, times, irradiations, amb_temps, predictions, weather_conditions, city, lat, lon)
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def display_results(self, forecast_data, times, irradiations, amb_temps, predictions, weather_conditions, city, lat, lon):
        current = forecast_data['list'][0]
        ttk.Label(self.current_frame, 
                 text=f"📍 Location: {city} ({lat:.2f}°N, {lon:.2f}°E)",
                 font=('Helvetica', 10, 'bold')).pack(anchor="w")
        ttk.Label(self.current_frame, text=f"⏰ Current Time: {datetime.fromtimestamp(current['dt']).strftime('%Y-%m-%d %H:%M')}").pack(anchor="w")
        ttk.Label(self.current_frame, text=f"Sunrise: {datetime.fromtimestamp(forecast_data['city']['sunrise']).strftime('%H:%M')} | Sunset: {datetime.fromtimestamp(forecast_data['city']['sunset']).strftime('%H:%M')}").pack(anchor="w")
        ttk.Label(self.current_frame, text=f"Current Temperature: {current['main']['temp']:.1f}°C").pack(anchor="w")
        ttk.Label(self.current_frame, text=f"Current Conditions: {current['weather'][0]['description']}").pack(anchor="w")
        
        self.ax.clear()
        time_labels = [t.strftime('%H:%M') for t in times]
        
        bars = self.ax.bar(time_labels, predictions, color=['orange' if p > 0 else 'gray' for p in predictions])
        self.ax.set_ylabel('Power (W)', color='orange')
        self.ax.set_xlabel('Time')
        self.ax.set_title('Solar Power Forecast (Next 6 Hours)')
        
        ax2 = self.ax.twinx()
        ax2.plot(time_labels, amb_temps, 'b-', marker='o', label='Temp (°C)')
        ax2.plot(time_labels, irradiations, 'g-', marker='s', label='Irradiation (W/m²)')
        ax2.set_ylabel('Temp / Irradiation')
        ax2.legend(loc='upper right')
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                self.ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
        
        self.canvas.draw()
        
        max_power = max(predictions)
        total_production = sum(predictions)
        daylight_hours = sum(1 for p in predictions if p > 0)
        avg_irradiation = np.mean(irradiations)
        
        # Alerts
        if daylight_hours == 0:
            self.alert_text.insert(tk.END, "🌙 NIGHTTIME\n• No solar production expected\n", "alert")
        else:
            if "rain" in [w.lower() for w in weather_conditions]:
                self.alert_text.insert(tk.END, "🌧️ RAIN\n• Expect reduced solar output due to rain\n", "alert")
            if "cloud" in [w.lower() for w in weather_conditions] and avg_irradiation < 300:
                self.alert_text.insert(tk.END, "☁️ CLOUDY\n• Cloud cover significantly affects irradiation\n", "alert")
            if total_production < 1000:
                self.alert_text.insert(tk.END, "🔋 LOW YIELD ->  Insufficient generation. Prepare for backup sources\n", "alert")
            if avg_irradiation > 600:
                self.alert_text.insert(tk.END, "🛰️ IoT ACTION SUGGESTED ->  Automate switching to solar for high efficiency\n", "alert")
        
        if daylight_hours == 0:
            self.recommend_text.insert(tk.END, "• Switch to grid/battery power\n", "recommend")
        else:
            peak_time = time_labels[np.argmax(predictions)]
            self.recommend_text.insert(tk.END, f"• Peak: {peak_time} ({max_power:.0f} W)\n", "recommend")
            if total_production > 2000:
                self.recommend_text.insert(tk.END, "• High production expected!\n", "recommend")
                self.recommend_text.insert(tk.END, "• Good time to charge batteries\n", "recommend")
            elif avg_irradiation < 200:
                self.alert_text.insert(tk.END, "• Poor conditions: reduce non-essential loads\n", "alert")
                

if __name__ == "__main__":
    root = tk.Tk()
    app = SolarPredictorApp(root)
    root.mainloop()
