import os
import zipfile
from pathlib import Path

def extract_all_zip_files(folder_path, password=None):
    """
    Extract all zip files in the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing zip files
        password (str): Password for encrypted zip files
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Find all zip files in the folder
    zip_files = list(folder_path.glob("*.zip"))
    
    if not zip_files:
        print(f"No zip files found in '{folder_path}'")
        return
    
    print(f"Found {len(zip_files)} zip file(s) to extract:")
    
    for zip_file in zip_files:
        print(f"\nExtracting: {zip_file.name}")
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Create extraction folder with same name as zip file (without .zip extension)
                extract_folder = folder_path / zip_file.stem
                extract_folder.mkdir(exist_ok=True)
                
                # Set password if provided
                if password:
                    zip_ref.setpassword(password.encode('utf-8'))
                
                # Extract all files
                zip_ref.extractall(extract_folder)
                print(f"✓ Successfully extracted to: {extract_folder}")
                
        except zipfile.BadZipFile:
            print(f"✗ Error: '{zip_file.name}' is not a valid zip file or is corrupted.")
        except RuntimeError as e:
            if "Bad password" in str(e):
                print(f"✗ Error: Wrong password for '{zip_file.name}'")
            else:
                print(f"✗ Error extracting '{zip_file.name}': {e}")
        except Exception as e:
            print(f"✗ Unexpected error with '{zip_file.name}': {e}")

if __name__ == "__main__":
    # Folder containing zip files
    dataset_folder = r"C:\Users\tungd\OneDrive - MSFT\Second Year\ML\ACMMM25 - Grand Challenge on Multimedia Verification\dataset\validation"
    
    # Password for encrypted zip files
    zip_password = "MultimediaVerification@ACMMM2025"
    
    print("Starting zip file extraction...")
    extract_all_zip_files(dataset_folder, zip_password)
    print("\nExtraction process completed.")