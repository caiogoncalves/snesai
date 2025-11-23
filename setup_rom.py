#!/usr/bin/env python3
"""
Setup script to extract and import Super Mario World ROM into gym-retro
"""
import os
import zipfile
import subprocess
import sys

def main():
    # Paths
    project_dir = "/Users/caio.goncalves/projetos/snesai"
    rom_zip = os.path.join(project_dir, "Super Mario World (U) [!].zip")
    rom_dir = os.path.join(project_dir, "roms")
    
    # Create roms directory if it doesn't exist
    os.makedirs(rom_dir, exist_ok=True)
    
    print("=" * 60)
    print("Super Mario World ROM Setup for gym-retro")
    print("=" * 60)
    
    # Extract ROM
    print(f"\n1. Extracting ROM from {rom_zip}...")
    try:
        with zipfile.ZipFile(rom_zip, 'r') as zip_ref:
            zip_ref.extractall(rom_dir)
        print("   ✓ ROM extracted successfully!")
        
        # List extracted files
        extracted_files = os.listdir(rom_dir)
        print(f"   Extracted files: {extracted_files}")
        
    except Exception as e:
        print(f"   ✗ Error extracting ROM: {e}")
        return 1
    
    # Import ROM into retro
    print(f"\n2. Importing ROM into stable-retro...")
    print("   Note: stable-retro may require manual ROM import.")
    print(f"   ROM location: {rom_dir}")
    
    try:
        # For stable-retro, we can try the import command
        result = subprocess.run(
            [sys.executable, "-m", "retro.import", rom_dir],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("   ✓ ROM imported successfully!")
            if result.stdout:
                print(f"   Output: {result.stdout}")
        else:
            print(f"   ⚠ Automatic import may have failed.")
            print(f"   Error: {result.stderr}")
            print(f"\n   Manual import instructions:")
            print(f"   Run: python -m retro.import {rom_dir}")
            # Don't return error, continue to verification
            
    except Exception as e:
        print(f"   ⚠ Could not run automatic import: {e}")
        print(f"\n   Manual import instructions:")
        print(f"   Run: python -m retro.import {rom_dir}")
    
    # Verify the import
    print("\n3. Verifying installation...")
    try:
        import retro
        games = retro.data.list_games()
        print(f"   Available games in retro: {len(games)} total")
        
        # Check if Super Mario World is available
        mario_games = [g for g in games if 'Mario' in g or 'SuperMarioWorld' in g]
        if mario_games:
            print(f"   ✓ Found Mario games: {mario_games}")
        else:
            print("   ⚠ Warning: No Mario games found in retro library")
            print(f"   All available games: {games}")
            print(f"\n   The ROM has been extracted to: {rom_dir}")
            print(f"   You may need to manually copy it to retro's data directory.")
            
    except Exception as e:
        print(f"   ⚠ Could not verify (retro may not be installed yet): {e}")
        print(f"   Please run: pip install -r requirements.txt")
    
    print("\n" + "=" * 60)
    print("Setup complete! You can now run: python train_mario.py")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
