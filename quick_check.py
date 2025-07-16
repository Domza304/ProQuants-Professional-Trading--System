"""
Quick Dependency Check for ProQuants
"""

print("ProQuants Dependency Check")
print("-" * 30)

# Critical dependencies
critical = [
    'pandas', 'numpy', 'MetaTrader5', 'tkinter'
]

# AI/ML dependencies  
ai_ml = [
    'sklearn', 'tensorflow', 'scipy'
]

missing_critical = []
missing_ai = []

for pkg in critical:
    try:
        if pkg == 'tkinter':
            import tkinter
        else:
            __import__(pkg)
        print(f"✓ {pkg}")
    except ImportError:
        print(f"✗ {pkg} - MISSING")
        missing_critical.append(pkg)

for pkg in ai_ml:
    try:
        __import__(pkg)
        print(f"✓ {pkg}")
    except ImportError:
        print(f"⚠ {pkg} - Missing (AI features disabled)")
        missing_ai.append(pkg)

print("\nSUMMARY:")
if missing_critical:
    print(f"❌ CRITICAL MISSING: {missing_critical}")
    print("Install with: pip install " + " ".join(missing_critical))
else:
    print("✅ All critical dependencies available")

if missing_ai:
    print(f"⚠ AI MISSING: {missing_ai}")
    print("Install with: pip install " + " ".join(missing_ai))
else:
    print("🧠 All AI dependencies available")
