#!/usr/bin/env python3
"""
Fix Unicode characters in professional_dashboard.py
"""

def fix_unicode_chars():
    try:
        # Read the file
        with open('src/gui/professional_dashboard.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace problematic Unicode box drawing characters with regular ASCII
        replacements = {
            '║': '|',
            '╔': '+',
            '╗': '+', 
            '╚': '+',
            '╝': '+',
            '╠': '+',
            '╣': '+',
            '═': '=',
            '├': '+',
            '┤': '+',
            '┬': '+',
            '┴': '+',
            '┼': '+',
            '─': '-',
            '└': '+',
            '┘': '+',
            '┌': '+',
            '┐': '+'
        }
        
        for unicode_char, ascii_char in replacements.items():
            content = content.replace(unicode_char, ascii_char)
        
        # Write back the fixed content
        with open('src/gui/professional_dashboard.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed Unicode characters in professional_dashboard.py")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing Unicode characters: {e}")
        return False

if __name__ == "__main__":
    fix_unicode_chars()
