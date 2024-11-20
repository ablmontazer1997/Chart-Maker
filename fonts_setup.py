# fonts_setup.py
import os
import shutil

def setup_fonts():
    """Setup Persian fonts for the application"""
    # Define font files
    fonts = [
        "BNazanin.ttf",
        "IRANYekanXFaNum-Medium.ttf",
        "Inter_24pt-Medium.ttf"
    ]
    
    # Ensure fonts directory exists
    font_dir = "fonts"
    if not os.path.exists(font_dir):
        os.makedirs(font_dir)
    
    # Print status of fonts
    print("Checking fonts:")
    for font in fonts:
        font_path = os.path.join(font_dir, font)
        if os.path.exists(font_path):
            print(f"✓ {font} is present")
        else:
            print(f"✗ {font} is missing")

def verify_fonts():
    """Verify that all required fonts are available"""
    font_dir = "fonts"
    required_fonts = [
        "BNazanin.ttf",
        "IRANYekanXFaNum-Medium.ttf",
        "Inter_24pt-Medium.ttf"
    ]
    
    missing_fonts = []
    for font in required_fonts:
        if not os.path.exists(os.path.join(font_dir, font)):
            missing_fonts.append(font)
    
    return len(missing_fonts) == 0, missing_fonts

if __name__ == "__main__":
    setup_fonts()
    success, missing = verify_fonts()
    if success:
        print("\nAll required fonts are available!")
    else:
        print("\nMissing fonts:")
        for font in missing:
            print(f"- {font}")
