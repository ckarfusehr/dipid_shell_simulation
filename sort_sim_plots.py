import os
import re
from pathlib import Path
import shutil
from PyPDF2 import PdfMerger

def sort_plots_into_folders(input_folder, output_folder):
    # Regex patterns (make `_random` optional)
    a_pattern = re.compile(r"_a([0-9.]+)")
    random_chance_pattern = re.compile(r"_random([0-9.]+)")

    # Ensure input and output folders exist
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if not input_folder.exists():
        print(f"Input folder {input_folder} does not exist.")
        return

    if not output_folder.exists():
        print(f"Output folder {output_folder} does not exist. Creating it.")
        output_folder.mkdir(parents=True, exist_ok=True)

    # Track created folders
    created_folders = []

    # Iterate over all files in the input folder
    for filepath in input_folder.glob("*.pdf"):
        filename_str = filepath.name
        print(f"Processing file: {filename_str}")

        # Extract `_a` and optionally `_random_chance` from the filename
        a_match = a_pattern.search(filename_str)
        random_chance_match = random_chance_pattern.search(filename_str)

        if not a_match:
            print(f"Skipping file {filename_str}: 'a' parameter not found.")
            continue

        # Get the extracted values
        a_value = a_match.group(1)
        random_chance_value = random_chance_match.group(1) if random_chance_match else "no_random"

        # Create a folder name based on `a` and `random_chance`
        folder_name = f"{a_value}_{random_chance_value}"
        folder_path = output_folder / folder_name

        # Create the folder if it doesn't exist
        folder_path.mkdir(parents=True, exist_ok=True)
        if folder_path not in created_folders:
            created_folders.append(folder_path)

        # Move the file into the folder
        target_path = folder_path / filepath.name
        shutil.move(filepath, target_path)

        print(f"Moved {filename_str} to {folder_path}")


def combine_pdfs_in_folder(folder_path, combined_pdfs_folder):
    # Initialize the PDF merger
    merger = PdfMerger()

    # Collect all individual PDF files in the folder
    pdf_files = sorted(f for f in folder_path.glob("*.pdf") if not f.name.endswith("_combined.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {folder_path}, skipping combination.")
        return None

    print(f"Combining PDFs in folder: {folder_path}")

    # Add each PDF to the merger
    for pdf_file in pdf_files:
        print(f"Adding {pdf_file} to the merger.")
        merger.append(str(pdf_file))

    # Save the combined PDF
    combined_pdf_name = folder_path.name + "_combined.pdf"
    combined_pdf_path = combined_pdfs_folder / combined_pdf_name
    merger.write(str(combined_pdf_path))
    merger.close()

    print(f"Combined PDF created: {combined_pdf_path}")
    return combined_pdf_path

def combine_all_merged_pdfs(combined_pdfs_folder, output_folder):
    # Initialize the PDF merger
    merger = PdfMerger()

    # Collect all merged PDFs, sorted by ascending `_a` and `_random_chance`
    combined_pdfs = []
    for combined_pdf_path in combined_pdfs_folder.glob("*_combined.pdf"):
        try:
            # Extract the folder name without "_combined"
            folder_name = combined_pdf_path.stem.replace("_combined", "")
            parts = folder_name.split("_")
            
            # Handle `_a` and `random_chance` parsing
            if len(parts) == 2:
                a_value, random_chance_value = parts
                if random_chance_value == "no_random":
                    random_chance_value = float("inf")
                else:
                    random_chance_value = float(random_chance_value.rstrip("."))
                combined_pdfs.append((float(a_value), random_chance_value, combined_pdf_path))
            else:
                print(f"Skipping file {combined_pdf_path}: invalid naming format.")
        except ValueError as e:
            print(f"Skipping file {combined_pdf_path}: invalid naming format. Error: {e}")

    # Sort by `_a` and `_random_chance`
    combined_pdfs.sort(key=lambda x: (x[0], x[1]))

    # Add each combined PDF to the merger
    for _, _, pdf_path in combined_pdfs:
        print(f"Adding {pdf_path} to the final merger.")
        merger.append(str(pdf_path))

    # Save the final combined PDF
    final_pdf_path = Path(output_folder) / "all_sim_images.pdf"
    if combined_pdfs:  # Check to avoid saving an empty PDF
        merger.write(str(final_pdf_path))
        merger.close()
        print(f"Final combined PDF created: {final_pdf_path}")
    else:
        print("No merged PDFs found to create the final combined PDF.")
        
        
if __name__ == "__main__":
    # Specify the input folder containing the simulation files
    input_folder = "sim_images"

    # Specify the output folder where sorted files should be stored
    output_folder = "sim_images/sorted_sim_images"

    # Sort the plots into folders and create merged PDFs
    sort_plots_into_folders(input_folder, output_folder)
    
    # Specify the output folder where sorted files are stored
    output_folder = "sim_images/sorted_sim_images"

    # Specify the folder to store all combined PDFs
    combined_pdfs_folder = Path("sim_images/combined_pdfs")
    combined_pdfs_folder.mkdir(parents=True, exist_ok=True)

    # Step 1: Merge PDFs in each subfolder and move to combined_pdfs_folder
    for subfolder in Path(output_folder).iterdir():
        if subfolder.is_dir():
            combine_pdfs_in_folder(subfolder, combined_pdfs_folder)

    # Step 2: Combine all merged PDFs into a single PDF
    combine_all_merged_pdfs(combined_pdfs_folder, combined_pdfs_folder)
