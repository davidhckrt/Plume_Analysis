import tkinter as tk
from tkinter import ttk
import numpy as np

class FOVCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("FOV Calculator")
        
        # Create main frame
        self.control_frame = ttk.Frame(root, padding="10")
        self.control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize parameters with default values
        self.params = {
            'fov_degrees': tk.DoubleVar(value=45.0),
            'distance_meters': tk.DoubleVar(value=10.0),
            'object_height_meters': tk.DoubleVar(value=2.0),
            'object_height_pixels': tk.DoubleVar(value=340),
            'image_height_pixels': tk.DoubleVar(value=350)
        }
        
        # Initialize which parameters are fixed
        self.fixed_params = {
            'fov_degrees': tk.BooleanVar(value=False),
            'distance_meters': tk.BooleanVar(value=False),
            'object_height_meters': tk.BooleanVar(value=False),
            'object_height_pixels': tk.BooleanVar(value=True),
            'image_height_pixels': tk.BooleanVar(value=True)
        }
        
        self.create_controls()
        
    def create_controls(self):
        # Create parameter input fields and checkboxes
        labels = {
            'fov_degrees': 'FOV (degrees)',
            'distance_meters': 'Distance (meters)',
            'object_height_meters': 'Object Height (meters)',
            'object_height_pixels': 'Object Height (pixels)',
            'image_height_pixels': 'Image Height (pixels)'
        }
        
        ranges = {
            'fov_degrees': (1, 180),
            'distance_meters': (0.1, 200),
            'object_height_meters': (0.1, 50),
            'object_height_pixels': (1, 1080),
            'image_height_pixels': (1, 1080)
        }
        
        row = 0
        for param, label in labels.items():
            ttk.Label(self.control_frame, text=label).grid(row=row, column=0, sticky=tk.W)
            
            # Entry field
            entry = ttk.Entry(self.control_frame, textvariable=self.params[param], width=10)
            entry.grid(row=row, column=1, padx=5)
            
            # Fixed checkbox
            ttk.Checkbutton(self.control_frame, text="Fixed", 
                           variable=self.fixed_params[param]).grid(row=row, column=2)
            
            # Range label
            ttk.Label(self.control_frame, 
                     text=f"Range: {ranges[param][0]} - {ranges[param][1]}").grid(row=row, column=3, sticky=tk.W)
            row += 1
        
        # Add calculate button
        ttk.Button(self.control_frame, text="Calculate", 
                  command=self.calculate_and_display).grid(row=row, column=0, columnspan=2, pady=10)
        
        # Add result display
        self.result_text = tk.Text(self.control_frame, height=4, width=40)
        self.result_text.grid(row=row+1, column=0, columnspan=4, pady=10)
        
    def calculate_missing_parameter(self):
        """Calculate the one parameter that is not fixed"""
        fov = self.params['fov_degrees'].get()
        d = self.params['distance_meters'].get()
        h_m = self.params['object_height_meters'].get()
        h_p = self.params['object_height_pixels'].get()
        i_h = self.params['image_height_pixels'].get()
        
        # Count unfixed parameters
        unfixed = []
        for param, fixed in self.fixed_params.items():
            if not fixed.get():
                unfixed.append(param)
        
        if len(unfixed) != 1:
            return "Please fix all but one parameter"
        
        unfixed_param = unfixed[0]
        fov_rad = np.radians(fov)
        
        try:
            if unfixed_param == 'fov_degrees':
                result = np.degrees(2 * np.arctan((h_m * i_h) / (2 * d * h_p)))
                self.params['fov_degrees'].set(result)
            elif unfixed_param == 'distance_meters':
                result = (h_m * i_h) / (2 * h_p * np.tan(fov_rad/2))
                self.params['distance_meters'].set(result)
            elif unfixed_param == 'object_height_meters':
                result = (2 * d * h_p * np.tan(fov_rad/2)) / i_h
                self.params['object_height_meters'].set(result)
            elif unfixed_param == 'object_height_pixels':
                result = (h_m * i_h) / (2 * d * np.tan(fov_rad/2))
                self.params['object_height_pixels'].set(result)
            elif unfixed_param == 'image_height_pixels':
                result = (2 * d * h_p * np.tan(fov_rad/2)) / h_m
                self.params['image_height_pixels'].set(result)
                
            return f"Calculated {unfixed_param}: {result:.2f}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"
        
    def calculate_and_display(self):
        result = self.calculate_missing_parameter()
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

def main():
    root = tk.Tk()
    app = FOVCalculator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
