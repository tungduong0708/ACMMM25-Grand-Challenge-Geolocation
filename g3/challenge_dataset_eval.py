import os
import csv
import base64
import torch
from pathlib import Path
import pandas as pd
from PIL import Image

# Import from the g3 module
from g3.g3_simple_prediction import G3Predictor, image_to_base64
from utils.utils import get_location_from_gps


def process_images_and_predict():
    """
    Process images from the clip_keyframes folder, run G3 prediction,
    get location information, and save results to CSV.
    """
    
    # Define paths
    image_folder = Path("g3/data/test/clip_keyframes/Video ID244")
    output_csv = Path("g3/data/test/clip_keyframes/prediction_results.csv")
    
    # Create output directory if it doesn't exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if image folder exists
    if not image_folder.exists():
        print(f"Error: Image folder {image_folder} does not exist!")
        return
    
    # Initialize G3 predictor
    print("Initializing G3 predictor...")
    try:
        predictor = G3Predictor(
            checkpoint_path="g3/checkpoints/mercator_finetune_weight.pth",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            index_path="g3/index/G3.index",
        )
        print("✅ G3 predictor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize G3 predictor: {e}")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(image_folder.glob(f"*{ext}"))
        image_files.extend(image_folder.glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images to process")
    
    if not image_files:
        print("No image files found in the specified folder!")
        return
    
    # CSV header
    csv_headers = ['image_path', 'latitude', 'longitude', 'location', 'status']
    
    # Initialize CSV file (write header if file doesn't exist)
    csv_exists = output_csv.exists()
    
    # Process each image
    processed_count = 0
    failed_count = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n📷 Processing image {i}/{len(image_files)}: {image_path.name}")
        
        try:
            # Convert image to base64
            print("  Converting image to base64...")
            base64_image = image_to_base64(str(image_path))
            
            # Run G3 prediction
            print("  Running G3 prediction...")
            predicted_gps = predictor.predict(
                base64_image=base64_image,
                database_csv_path="g3/data/coordinates_100K.csv",  # Adjust path as needed
                top_k=20,
                model_name="gemini-2.5-pro",
            )
            
            latitude, longitude = predicted_gps
            print(f"  Predicted GPS: ({latitude:.6f}, {longitude:.6f})")
            
            # Get location from GPS coordinates
            print("  Getting location information...")
            location = get_location_from_gps(latitude, longitude)
            print(f"  Location: {location}")
            
            # Prepare data for CSV
            row_data = [
                str(image_path),
                float(latitude),
                float(longitude),
                location,
                'success'
            ]
            
            processed_count += 1
            status = 'success'
            
        except Exception as e:
            print(f"  ❌ Error processing {image_path.name}: {e}")
            
            # Still save the row with error information
            row_data = [
                str(image_path),
                None,
                None,
                f"Error: {str(e)}",
                'failed'
            ]
            
            failed_count += 1
            status = 'failed'
        
        # Append to CSV file
        try:
            with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header if this is the first row
                if not csv_exists:
                    writer.writerow(csv_headers)
                    csv_exists = True
                
                writer.writerow(row_data)
            
            print(f"  ✅ Results saved to CSV (Status: {status})")
            
        except Exception as e:
            print(f"  ❌ Failed to save to CSV: {e}")
    
    # Final summary
    print(f"\n📊 Processing Summary:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Results saved to: {output_csv}")
    
    # Display first few results
    if output_csv.exists():
        try:
            df = pd.read_csv(output_csv)
            print(f"\n📋 Sample Results (first 5 rows):")
            print(df.head().to_string(index=False))
        except Exception as e:
            print(f"Could not display results: {e}")


def validate_setup():
    """
    Validate that all required files and directories exist.
    """
    print("🔍 Validating setup...")
    
    required_files = [
        "g3/checkpoints/mercator_finetune_weight.pth",
        "g3/index/G3.index",
        "g3/data/coordinates_100K.csv",
    ]
    
    required_dirs = [
        "g3/data/test/clip_keyframes/ID31-1",
    ]
    
    all_good = True
    
    # Check required files
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"  ❌ Missing required file: {file_path}")
            all_good = False
        else:
            print(f"  ✅ Found: {file_path}")
    
    # Check required directories
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"  ❌ Missing required directory: {dir_path}")
            all_good = False
        else:
            print(f"  ✅ Found: {dir_path}")
    
    # Check environment variable
    api_key = os.getenv("API_KEY")
    if not api_key:
        print(f"  ❌ Missing environment variable: API_KEY")
        all_good = False
    else:
        print(f"  ✅ API_KEY is set")
    
    if all_good:
        print("✅ All requirements satisfied!")
    else:
        print("❌ Some requirements are missing. Please check the above issues.")
    
    return all_good


if __name__ == "__main__":
    print("🚀 Starting Image Processing and G3 Prediction Pipeline")
    print("=" * 60)
    
    # Validate setup first
    if not validate_setup():
        print("\n❌ Setup validation failed. Please fix the issues above.")
        exit(1)
    
    print("\n🎯 Starting image processing...")
    process_images_and_predict()
    
    print("\n🎉 Pipeline completed!")
