import sys
import os

print("Python Environment Test")
print("=" * 30)
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print(f"Current Directory: {os.getcwd()}")
print(f"Platform: {sys.platform}")

# Test tkinter
try:
    import tkinter as tk
    print("✓ Tkinter imported successfully")
    
    # Test basic window
    root = tk.Tk()
    root.title("Test")
    root.geometry("200x100")
    
    label = tk.Label(root, text="GUI Working!")
    label.pack(pady=20)
    
    print("✓ Test window created")
    
    # Auto-close
    root.after(2000, root.destroy)
    root.mainloop()
    
    print("✓ GUI test completed successfully")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
